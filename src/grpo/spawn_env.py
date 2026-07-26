"""Safely build LIBERO async vec envs in the spawn context.

Forking from a CUDA-initialized process deadlocks (measured: the 3rd env
build in GRPO hung for 60 minutes, with the "multi-threaded, use of fork()
may lead to deadlocks" warning). gymnasium's AsyncVectorEnv(context="spawn")
starts workers in a fresh interpreter, but requires env_fn to be picklable.
The fn returned by lerobot's create_libero_envs is a closure and cannot be
pickled, so this module provides a module-level factory taking only
primitive arguments.

Spawn worker startup is slower than fork (~10-30 s for interpreter + import),
but the G workers start in parallel, so it costs only about +1 minute per
env build.
"""
from __future__ import annotations

from functools import partial

import gymnasium as gym


class RenderSkipWrapper(gym.Wrapper):
    """Pause camera observables to skip wasted rendering during an action chunk.

    SmolVLA emits n_action_steps(=50) actions per inference, but only one
    observation — the one after the chunk finishes — feeds the next inference.
    robosuite renders every camera at every step by default, so 49/50 frames
    are thrown away. set_render_skip(True) sets the camera observables to
    enabled=False, which stops the render computation (active stays True, so a
    stale image remains in the obs dict and upstream observation shaping does
    not break); switching back to False before the chunk's final step triggers
    exactly one fresh render there. Physics, success checks, and the staged
    reward are sim-state based and unaffected. That is the design.

    **It does not actually engage under gymnasium 1.x (re-audit, 2026-07).**
    `_cameras()` below walks the wrapper stack looking for robosuite's
    `_observables`, but gymnasium 1.x removed `Wrapper.__getattr__`, so the probe
    does not resolve through RenderSkipWrapper(StagedRewardWrapper(LiberoEnv)).
    It falls back to an empty camera list and `set_render_skip` returns 0, having
    stopped nothing. check_render_skip.py passes in that state because a no-op
    also produces identical observations, so **it is not proof of equivalence**.
    The measured x0.99 was explained as "EGL rendering runs on the GPU, so cutting
    renders buys nothing"; that explanation has been withdrawn. Both GRPO runs use
    render_skip: false, so nothing downstream depends on this.
    """

    def __init__(self, env):
        super().__init__(env)
        self._rs_env = None
        self._cam_names: list | None = None

    def _cameras(self) -> list:
        if self._cam_names is None:
            try:
                # LiberoEnv → ._env (OffScreenRenderEnv) → .env (robosuite env)
                inner = getattr(self.env, "_env", None) or getattr(self.env, "env", None) or self.env
                e = getattr(inner, "env", None) or inner
                obs_map = getattr(e, "_observables", None)
                if obs_map and hasattr(e, "modify_observable"):
                    self._rs_env = e
                    self._cam_names = [n for n in obs_map if n.endswith("_image")]
                else:
                    self._cam_names = []
            except Exception:  # noqa: BLE001  (probe unavailable -> degrade to always rendering)
                self._cam_names = []
        return self._cam_names

    def set_render_skip(self, skip: bool) -> int:
        """Stop (True) / resume (False) rendering. Returns the number of cameras controlled (for diagnostics)."""
        names = self._cameras()
        for n in names:
            try:
                self._rs_env.modify_observable(n, "enabled", not skip)
            except Exception:  # noqa: BLE001
                return 0
        return len(names)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # The underlying env may be rebuilt on reset, so re-probe and clear the
        # skip so that everything from the initial observation on is freshly rendered
        self._cam_names = None
        self._rs_env = None
        self.set_render_skip(False)
        return out


def _build_env(
    suite_name: str,
    task_id: int,
    episode_index: int,
    n_envs: int,
    episode_length,
    camera_name_mapping: dict | None,
    control_mode: str,
    gym_kwargs: dict,
    staged_reward: bool,
):
    """Env factory executed inside the worker process (all arguments are picklable)."""
    # Prevent BLAS/OpenMP thread explosion (if each of the G×workers processes
    # spawns its own thread pool, context switching slows everything down).
    # Must be set before the imports.
    import os

    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from lerobot.envs.libero import LiberoEnv

    try:
        from lerobot.envs.libero import _get_suite

        suite = _get_suite(suite_name)
    except ImportError:  # fallback in case the private API changed
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name]()

    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        episode_length=episode_length,
        episode_index=episode_index,
        n_envs=n_envs,
        camera_name_mapping=camera_name_mapping,
        control_mode=control_mode,
        is_libero_plus=True,
        **gym_kwargs,
    )
    if staged_reward:
        from src.grpo.privileged import StagedRewardWrapper

        env = StagedRewardWrapper(env)
    # Outermost shell: suppress wasted rendering during chunks (controlled via venv.call("set_render_skip", ...))
    return RenderSkipWrapper(env)


def make_spawn_vec_env(
    suite: str,
    task_id: int,
    camera_name_mapping: dict | None,
    n_envs: int,
    staged_reward: bool = False,
    episode_index_base: int = 0,
):
    """Vec env for one task. n_envs>1 uses async with the spawn context.

    episode_index_base: gives sub-env i an episode_index of base+i.

    IMPORTANT — in LIBERO-Plus the initial state CANNOT be chosen with a seed.
    LiberoEnv.reset() applies the seed and only *afterwards* calls
    set_init_state(init_states[init_state_id % N]), which overwrites whatever
    the seed produced; init_state_id starts from episode_index and advances by
    n_envs on every reset. The initial state is therefore determined solely by
    (episode_index, number of resets) — seeding has no effect on it.
    The default of 0 is exactly identical to the previous behaviour
    (rollout collection / GRPO training).

    **Shifting this base does NOT buy a held-out evaluation on a trained task**
    (analysis of 2026-07-25). A terminated episode triggers two resets
    (LiberoEnv.step's internal reset plus the vector env's autoreset), so the
    indices collection touches are outcome-dependent and scattered, reaching as
    high as 79, and they wrap via % N: at N=50 that blocks 32 of the 50 states and
    no free window of width 15 exists (enumerate with src/init_state_coverage.py).
    There are exactly two legitimate uses of this base:
      1. evaluating untrained tasks (no contamination is possible, so the offset
         can stay at 0)
      2. rebuilding the venv per wave with base = offset + wave*B, so that the
         initial state never depends on the arm's own success pattern
         (the paired samples in eval_heldout)

    Returns: (env_cfg, vec_env) — env_cfg is for building the policy/processor.
    """
    import gymnasium as gym

    from lerobot.envs.configs import LiberoPlusEnv

    env_cfg = LiberoPlusEnv(
        task=suite, task_ids=[task_id], camera_name_mapping=camera_name_mapping
    )
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)  # LiberoEnv does not accept it

    fns = [
        partial(
            _build_env,
            suite,
            task_id,
            ep_idx,
            n_envs,
            env_cfg.episode_length,
            camera_name_mapping,
            env_cfg.control_mode,
            gym_kwargs,
            staged_reward,
        )
        for ep_idx in range(episode_index_base, episode_index_base + n_envs)
    ]
    if n_envs == 1:
        return env_cfg, gym.vector.SyncVectorEnv(fns)
    # copy=False: skip the copy when returning shared-memory observations. The
    # observation is immediately re-tensorized by the preprocessor (float
    # conversion → CUDA transfer), and workers only write again at the next
    # venv.step, so aliasing is safe.
    return env_cfg, gym.vector.AsyncVectorEnv(fns, context="spawn", copy=False)
