"""Env wrapper that attaches a privileged-state staged reward to info.

The criteria of the diagnostic tool (src/diagnose_rollouts.py) are scored
directly as a reward:

    r = 0.15 * approach_frac        # progress toward target (continuous, 0-1)
      + 0.20 * reached              # got within 10 cm of the target
      + 0.20 * grasp_attempted      # gripper-close command at close range (6 cm)
      + 0.20 * lifted               # target z rose by ≥ 4 cm (only after grasp)
      + 0.25 * success              # env success check

Monotone gating (points can only accrue in reached→grasp→lift order) prevents
reward hacking. With binary success alone, an all-fail group has zero
advantage and the GRPO signal vanishes; this dense reward creates differences
within a group. The design follows lecture 4 (pp. 13-14: "RL excels at tasks
with dense rewards / define dense rewards to ease exploration"), and — like
the asymmetric actor-critic exercise (only the critic sees privileged
information) — uses simulator privileged state as scaffolding for exploration.

The wrapper runs inside each async vec env worker process and returns
info["staged_reward"] plus its components at episode end. If the privileged
probe is unavailable, it degrades gracefully to success-only reward (0 or 1).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

NEAR = 0.10
GRASP_NEAR = 0.06
LIFT_DZ = 0.04

W_APPROACH = 0.15
W_REACH = 0.20
W_GRASP = 0.20
W_LIFT = 0.20
W_SUCCESS = 0.25


class StagedRewardWrapper(gym.Wrapper):
    """Wraps LiberoEnv and computes the staged reward from privileged state inside the worker."""

    def __init__(self, env):
        super().__init__(env)
        self._probe_ready = False
        self._reset_track()

    # ---- Privileged probe (same path as diagnose_rollouts.SimProbe, in-worker version) ----
    def _ensure_probe(self):
        if self._probe_ready:
            return
        try:
            # LiberoEnv → ._env (OffScreenRenderEnv) → .env (libero robosuite env).
            # Also handles envs (e.g. in tests) where the attributes live directly on the object.
            inner = getattr(self.env, "_env", None) or getattr(self.env, "env", None) or self.env
            e = getattr(inner, "env", None) or inner
            self._sim_env = e
            self._obj_of_interest = list(getattr(e, "obj_of_interest", []) or [])
            self._obj_body_id = dict(getattr(e, "obj_body_id", {}) or {})
            robot = e.robots[0]
            sid = getattr(robot, "eef_site_id", None)
            if sid is None:
                sid = e.sim.model.site_name2id(f"{robot.gripper.naming_prefix}grip_site")
            self._eef_site_id = sid
            self._probe_ready = bool(self._obj_of_interest and self._obj_body_id)
        except Exception:  # noqa: BLE001  (probe unavailable -> degrade to success-only reward)
            self._probe_ready = False

    def _target_positions(self) -> dict:
        """Return {object name: current xyz} (a dict so each object keeps its identity)."""
        out = {}
        for name, bid in self._obj_body_id.items():
            if not any(t in name or name in t for t in self._obj_of_interest):
                continue
            if isinstance(bid, dict):
                bid = next(iter(bid.values()))
            out[name] = np.array(self._sim_env.sim.data.body_xpos[bid])
        return out

    def _reset_track(self):
        self._d0 = None
        self._min_d = np.inf
        self._z0_by_name = {}       # per-object z at reset (lift is judged against this)
        self._grasp_target = None   # lock the name of the grasped object (lift counts only for it)
        self._reached = False
        self._grasped = False
        self._lifted = False
        self._t = 0                 # in-episode step counter (for milestone timestamps)
        self._step_reached = -1     # step at which each stage was first reached (-1 = never)
        self._step_grasped = -1
        self._step_lifted = -1

    # ---- gym API ----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_track()
        self._probe_ready = False  # the underlying env may be rebuilt on reset
        # Take the initial distance d0 from the state right after reset (it would be underestimated after the first step)
        self._ensure_probe()
        if self._probe_ready:
            try:
                eef = np.array(self._sim_env.sim.data.site_xpos[self._eef_site_id])
                targets = self._target_positions()
                if targets:
                    self._d0 = max(
                        min(float(np.linalg.norm(eef - p)) for p in targets.values()), 1e-3
                    )
                    self._z0_by_name = {n: float(p[2]) for n, p in targets.items()}
            except Exception:  # noqa: BLE001
                self._probe_ready = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ensure_probe()
        if self._probe_ready:
            try:
                self._track(action)
            except Exception:  # noqa: BLE001
                self._probe_ready = False
        if terminated or truncated:
            info = dict(info)
            info.update(self._staged_reward(info))
        return obs, reward, terminated, truncated, info

    def _track(self, action):
        eef = np.array(self._sim_env.sim.data.site_xpos[self._eef_site_id])
        targets = self._target_positions()
        if not targets:
            return
        # Compute per-object distances and treat the nearest object as the
        # current target. If reach→grasp→lift are not tied to a single object,
        # reward hacking occurs: the policy approaches and commands a grasp on
        # A while B happens to rise, scoring grasp/lift points (caught by Codex).
        dists = {n: float(np.linalg.norm(eef - p)) for n, p in targets.items()}
        nearest = min(dists, key=dists.get)
        d = dists[nearest]
        if self._d0 is None:
            self._d0 = max(d, 1e-3)
        if not self._z0_by_name:
            self._z0_by_name = {n: float(p[2]) for n, p in targets.items()}
        self._min_d = min(self._min_d, d)
        self._t += 1
        if d < NEAR and not self._reached:
            self._reached = True
            self._step_reached = self._t
        gripper_close = bool(np.asarray(action).reshape(-1)[-1] > 0)
        if self._reached and gripper_close and d < GRASP_NEAR and not self._grasped:
            self._grasped = True
            self._step_grasped = self._t
            if self._grasp_target is None:
                self._grasp_target = nearest  # lock the grasped object
        # lift counts only when the locked object itself rises above its own initial z
        if self._grasped and self._grasp_target is not None and not self._lifted:
            z = float(targets[self._grasp_target][2])
            z0 = self._z0_by_name.get(self._grasp_target)
            if z0 is not None and z - z0 > LIFT_DZ:
                self._lifted = True
                self._step_lifted = self._t

    def staged_now(self) -> dict:
        """Return the staged reward from tracking state so far (works mid-episode).

        The LIBERO env never truncates failed episodes on its own (terminated
        only fires on success). When the rollout loop cuts off at the step
        limit from outside, step() never emits terminal info, so at cutoff we
        call this method directly via venv.call to collect partial credit.
        Treated as success=False (a success would already have been captured
        through the terminated=True path).
        """
        return self._staged_reward({})

    def _staged_reward(self, info) -> dict:
        success = bool(info.get("is_success", False))
        if self._d0 is None:  # probe unavailable -> success only
            return {
                "staged_reward": float(success),
                "staged_components": {},
                "staged_probe_ready": 0.0,
            }
        approach = float(np.clip(1.0 - self._min_d / self._d0, 0.0, 1.0))
        comp = {
            "approach": approach,
            # Note: "grasped" means "issued a gripper-close command at close range"
            # = grasp_attempted, NOT a contact-sensor-confirmed grasp (state this
            # explicitly in the paper and analyses)
            "reached": float(self._reached),
            "grasped": float(self._grasped),
            "lifted": float(self._lifted),
            "success": float(success),
        }
        r = (
            W_APPROACH * approach
            + W_REACH * comp["reached"]
            + W_GRASP * comp["grasped"]
            + W_LIFT * comp["lifted"]
            + W_SUCCESS * comp["success"]
        )
        out = {"staged_reward": float(r), "staged_components": comp}
        # Flat output for diagnostics: gymnasium 1.x aggregates info in a
        # "scalar + _key mask" format, so components/milestones are also
        # emitted as scalars (the rollout collects them per env)
        for k, v in comp.items():
            out[f"staged_{k}"] = float(v)
        out["staged_probe_ready"] = 1.0
        out["staged_min_dist"] = float(self._min_d if np.isfinite(self._min_d) else -1.0)
        out["staged_step_reached"] = float(self._step_reached)
        out["staged_step_grasped"] = float(self._step_grasped)
        out["staged_step_lifted"] = float(self._step_lifted)
        return out


def make_staged_vec_env(
    suite: str,
    task_id: int,
    camera_name_mapping: dict | None,
    n_envs: int,
):
    """Vec env with StagedRewardWrapper (wrapped inside the spawn worker)."""
    from .spawn_env import make_spawn_vec_env

    return make_spawn_vec_env(suite, task_id, camera_name_mapping, n_envs, staged_reward=True)
