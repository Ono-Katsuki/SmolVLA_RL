"""Mechanical organ diagnosis of rollouts (auto-classify from privileged state whether the brain/eyes/hand is at fault).

Instead of eyeballing videos, record MuJoCo privileged state (EEF position,
obj_of_interest object positions, gripper commands) at every step and
classify episodes by thresholds:

    frozen       total EEF travel is nearly zero          → plumbing (broken action path)
    no_approach  never approaches the target              → eyes/brain (perception/grounding)
    wrong_target significantly approaches a non-target    → brain/eyes
    near_miss    approaches but never attempts a grasp    → hand/eyes
    grasp_fail   closes the gripper at close range but fails to lift → hand (precision grasp)
    erratic      repeatedly approaches then retreats      → hand/coupling
    success      env success check

Usage:
    python -m src.diagnose_rollouts \
        --checkpoint <ckpt> --suite libero_spatial --task_ids '[51,228]' \
        --episodes 3 --output_dir /content/diag

Access to privileged state depends on LIBERO env internals, so the code is
defensive; fields that could not be obtained are reported in the summary's
"probe_fields" (even if everything fails, the motion-based classes
frozen/erratic still work).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Classification thresholds (meters)
NEAR = 0.10        # distance to target considered "approached"
GRASP_NEAR = 0.06  # distance at gripper closure considered a grasp attempt
LIFT_DZ = 0.04     # target z rise considered "lifted"
FROZEN_PATH = 0.05  # total travel below this is frozen
RETREAT_DELTA = 0.05  # moving this far away after closest approach counts as one retreat

ORGAN = {
    "frozen": "plumbing",
    "no_approach": "eyes/brain",
    "wrong_target": "brain/eyes",
    "near_miss": "hand/eyes",
    "grasp_fail": "hand",
    "erratic": "hand/coupling",
    "success": "-",
    "motion_directed": "hand (target probe unavailable)",
}


@dataclass
class EpisodeTrace:
    eef: list = field(default_factory=list)          # (3,) per step
    targets: list = field(default_factory=list)      # dict[name, (3,)] per step
    distractors: list = field(default_factory=list)  # dict[name, (3,)] per step
    gripper_close: list = field(default_factory=list)  # bool per step (command-based)
    success: bool = False
    n_steps: int = 0


# ------------------------------------------------------------
# Privileged-state probe (depends on LIBERO internals → defensive)
# ------------------------------------------------------------
class SimProbe:
    def __init__(self, venv):
        self.env = None
        self.obj_of_interest: list[str] = []
        self.obj_body_id: dict[str, int] = {}
        self.eef_site_id: int | None = None
        self.fields: dict[str, bool] = {"eef": False, "targets": False, "distractors": False}
        try:
            # SyncVectorEnv(n_envs=1) → LiberoEnv → OffScreenRenderEnv → libero env
            lerobot_env = venv.envs[0]
            inner = getattr(lerobot_env, "_env", None) or getattr(lerobot_env, "env", None)
            self.env = getattr(inner, "env", inner)
        except Exception as e:  # noqa: BLE001
            print(f"[diagnose] env unwrap failed: {e}")
            return
        e = self.env
        self.obj_of_interest = list(getattr(e, "obj_of_interest", []) or [])
        self.obj_body_id = dict(getattr(e, "obj_body_id", {}) or {})
        try:
            robot = e.robots[0]
            sid = getattr(robot, "eef_site_id", None)
            if sid is None:
                name = f"{robot.gripper.naming_prefix}grip_site"
                sid = e.sim.model.site_name2id(name)
            self.eef_site_id = sid
            self.fields["eef"] = True
        except Exception as ex:  # noqa: BLE001
            print(f"[diagnose] eef probe unavailable: {ex}")
        self.fields["targets"] = bool(self.obj_of_interest and self.obj_body_id)
        self.fields["distractors"] = bool(
            self.obj_body_id and len(self.obj_body_id) > len(self.obj_of_interest)
        )
        print(
            f"[diagnose] probe: eef={self.fields['eef']} "
            f"targets={sorted(self.obj_of_interest)} n_bodies={len(self.obj_body_id)}"
        )

    def _body_pos(self, name: str):
        bid = self.obj_body_id.get(name)
        if bid is None:
            return None
        # defensive: some implementations store dict values in obj_body_id
        if isinstance(bid, dict):
            bid = next(iter(bid.values()))
        return np.array(self.env.sim.data.body_xpos[bid])

    def sample(self) -> tuple[np.ndarray | None, dict, dict]:
        if self.env is None:
            return None, {}, {}
        eef = None
        if self.fields["eef"]:
            eef = np.array(self.env.sim.data.site_xpos[self.eef_site_id])
        targets, distractors = {}, {}
        for name in self.obj_body_id:
            pos = self._body_pos(name)
            if pos is None:
                continue
            if any(t in name or name in t for t in self.obj_of_interest):
                targets[name] = pos
            else:
                distractors[name] = pos
        return eef, targets, distractors


# ------------------------------------------------------------
# Classification (pure function — unit-testable)
# ------------------------------------------------------------
def classify_episode(trace: EpisodeTrace) -> dict:
    out: dict = {"n_steps": trace.n_steps, "success": trace.success}
    eef = np.array(trace.eef) if trace.eef else None
    if trace.success:
        out["label"] = "success"
        return _finish(out)

    if eef is None or len(eef) < 2:
        out["label"] = "frozen"
        return _finish(out)

    path_len = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
    out["path_len"] = round(path_len, 3)
    if path_len < FROZEN_PATH:
        out["label"] = "frozen"
        return _finish(out)

    has_targets = any(t for t in trace.targets)
    if not has_targets:
        # if target positions are unavailable, motion-based only: judge erratic by straightness
        net = float(np.linalg.norm(eef[-1] - eef[0]))
        out["directness"] = round(net / max(path_len, 1e-6), 3)
        out["label"] = "erratic" if out["directness"] < 0.15 else "motion_directed"
        return _finish(out)

    # distance-to-target series (using the nearest target)
    d = np.array(
        [
            min(np.linalg.norm(e - p) for p in t.values()) if t else np.inf
            for e, t in zip(eef, trace.targets)
        ]
    )
    min_d = float(np.min(d))
    out["min_dist_to_target"] = round(min_d, 3)

    # Retreat count: times the distance moved back out by ≥ RETREAT_DELTA from
    # the running minimum. Returning near the minimum (within Δ/2) clears
    # 'away' so oscillations keep being counted
    retreats = 0
    running_min = d[0]
    away = False
    for x in d:
        if x < running_min:
            running_min = x
            away = False
        elif away and x < running_min + RETREAT_DELTA / 2:
            away = False
        elif not away and x > running_min + RETREAT_DELTA:
            retreats += 1
            away = True
    out["retreats"] = retreats

    # grasp attempt: gripper-close command at close range to the target
    grasp_attempted = any(
        g and dist < GRASP_NEAR for g, dist in zip(trace.gripper_close, d)
    )
    out["grasp_attempted"] = grasp_attempted

    # lift: rise of the target's z
    z0 = min(p[2] for p in trace.targets[0].values())
    z_max = max(max(p[2] for p in t.values()) for t in trace.targets if t)
    out["lift_dz"] = round(float(z_max - z0), 3)

    # approach to non-targets
    dd = [
        min((np.linalg.norm(e - p) for p in t.values()), default=np.inf)
        for e, t in zip(eef, trace.distractors)
    ]
    min_dd = float(np.min(dd)) if dd else np.inf
    out["min_dist_to_distractor"] = None if np.isinf(min_dd) else round(min_dd, 3)

    if min_d < GRASP_NEAR and grasp_attempted:
        out["label"] = "grasp_fail"
    elif min_d < NEAR:
        out["label"] = "erratic" if retreats >= 2 else "near_miss"
    elif min_dd < min_d - 0.05:
        out["label"] = "wrong_target"
    elif retreats >= 2:
        out["label"] = "erratic"
    else:
        out["label"] = "no_approach"
    return _finish(out)


def _finish(out: dict) -> dict:
    out["organ"] = ORGAN[out["label"]]
    return out


# ------------------------------------------------------------
# Rollout execution
# ------------------------------------------------------------
def run(args) -> None:
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    from .grpo.train_grpo import (
        _build_processors,
        _camera_name_mapping,
        _load_policy,
        _make_task_env,
        _resolve_pretrained_dir,
    )

    import lerobot.policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_dir = _resolve_pretrained_dir(args.checkpoint)
    cam_map = _camera_name_mapping(PreTrainedConfig.from_pretrained(str(pretrained_dir)))
    task_ids = json.loads(args.task_ids)

    results: dict[str, list[dict]] = {}
    env_cfg = policy = processors = None
    for tid in task_ids:
        env_cfg_i, venv = _make_task_env(args.suite, int(tid), cam_map, n_envs=1)
        if policy is None:
            env_cfg = env_cfg_i
            policy_cfg, policy = _load_policy(pretrained_dir, env_cfg, device)
            processors = _build_processors(policy_cfg, env_cfg, pretrained_dir, device)
            policy.eval()
        probe = SimProbe(venv)
        per_task = []
        for ep in range(args.episodes):
            policy.reset()
            obs, info = venv.reset(seed=[args.seed + ep])
            trace = EpisodeTrace()
            done = False
            step = 0
            while not done and step < args.max_steps:
                eef, targets, distractors = probe.sample()
                observation = preprocess_observation(obs)
                try:
                    observation["task"] = list(venv.call("task_description"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = [""]
                observation = processors["env_preprocessor"](observation)
                observation = processors["preprocessor"](observation)
                with torch.inference_mode():
                    action = policy.select_action(observation)
                action = processors["postprocessor"](action)
                transition = processors["env_postprocessor"]({ACTION: action})
                action_np = transition[ACTION].to("cpu").numpy()

                if eef is not None:
                    trace.eef.append(eef)
                    trace.targets.append(targets)
                    trace.distractors.append(distractors)
                    # LIBERO's gripper command is the last action dim (>0 = close)
                    trace.gripper_close.append(bool(action_np[0, -1] > 0))

                obs, _r, term, trunc, info = venv.step(action_np)
                step += 1
                done = bool(np.logical_or(term, trunc).reshape(-1)[0])
            trace.n_steps = step
            fi = info.get("final_info") if isinstance(info, dict) else None
            if isinstance(fi, dict):
                trace.success = bool(np.asarray(fi.get("is_success", [False])).reshape(-1)[0])
            elif fi is not None:
                trace.success = any(
                    isinstance(x, dict) and x.get("is_success") for x in fi
                )
            per_task.append(classify_episode(trace))
            print(f"[diagnose] {args.suite}:{tid} ep{ep}: {per_task[-1]}")
        results[f"{args.suite}:{tid}"] = per_task
        venv.close()

    labels = [r["label"] for rs in results.values() for r in rs]
    summary = {
        "checkpoint": str(args.checkpoint),
        "counts": {l: labels.count(l) for l in sorted(set(labels))},
        "per_task": results,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["counts"], indent=2))
    print(f"[diagnose] wrote {out / 'diagnosis.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_ids", required=True, help='JSON array (e.g. "[51,228]")')
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
