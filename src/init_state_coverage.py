"""Reconstruct which LIBERO init-state indices each training stage actually visited.

WHY THIS EXISTS
---------------
LIBERO-Plus does not choose the initial state from the reset seed. `LiberoEnv.reset`
calls `set_init_state(init_states[init_state_id % N])` *after* seeding, where
`init_state_id` starts at the sub-env's `episode_index` and advances by `n_envs`
on every reset. Crucially a *terminated* episode causes AT LEAST TWO resets
(LiberoEnv resets internally on termination, and the vector env autoresets on the
next step). So the set of visited indices depends on the OUTCOME PATTERN, not just
on the episode count.

"At least" and "outcome", not "exactly" and "success", both matter (2026-07-29):
after the autoreset the slot is running a fresh episode and is still being
stepped -- the collection loop runs until *every* slot is done -- so it can
terminate again and consume more indices. And `terminated = done or is_success`
leaves the underlying env's `done` as a second sufficient cause, so termination
is not synonymous with success.

That makes "the training rollouts used states {0..31}" wrong whenever a rollout
succeeded: a success in round 0 sends that sub-env to 24, 32, 40 instead of
8, 16, 24. Picking an evaluation offset by assuming a contiguous {0..31} block
therefore silently re-evaluates states the policy was trained on -- which is the
exact bug this module exists to prevent us from repeating.

USAGE
-----
    python -m src.init_state_coverage --episodes <collect>/episodes.jsonl \
        --group-size 8 --rounds 4 --n-episodes 15 [--n-init 50]

Prints, per task, the indices collection actually visited, the GRPO training
indices, and the smallest evaluation offset whose `--n-episodes` window is
disjoint from both. Exits non-zero if no such offset exists, because in that
case the offset approach cannot deliver a held-out evaluation on that task and
the honest move is to say so rather than run.

WHAT IT REPORTED FOR THIS PROJECT (2026-07-25), AND HOW FAR TO TRUST IT
-----------------------------------------------------------------------
No offset works. Collection reaches scattered indices as high as 79 rather than
a clean {0..31}, and once those wrap modulo the real N=50 the blocked set covers
32 of the 50 states a typical LIBERO-Plus task provides -- there is no free
contiguous window of the width (15) eval A needed.

**Those specific numbers are an unvalidated estimate, not an established fact
(corrected 2026-07-29).** They come out of `collection_visited` below, which
assumes exactly one termination per round followed by exactly one autoreset. That
assumption can fail in BOTH directions -- see that function's docstring -- so the
reconstructed set is neither a subset nor a superset of the set actually visited.
Settling it would take real per-reset initial-state traces, which were never
recorded. (That gap is itself why the upstream report asks for the initial-state
index to be exposed per episode: huggingface/lerobot#4152.)

What does not depend on the arithmetic, and is enough on its own: which states
collection touched is outcome-dependent and cannot be reconstructed reliably from
the success bits we kept. An offset therefore cannot be *shown* to be held out,
which is the only thing the decision below ever needed.

So this module is a diagnostic that closed the offset route, not a recipe for
using it: do not read the `USE --init_state_offset ...` line below as advice that
the route is open in general. It prints only in the case where a disjoint window
happens to exist -- and given the above, "disjoint" there means "disjoint from the
reconstruction", which is weaker than it sounds.

The remedy actually adopted was to evaluate tasks that were never trained on, so
that no initial state can be contaminated by construction, and to rebuild the
vector env per wave so the pairing cannot depend on outcomes. See
`data/eval/PREREGISTRATION_unseen_task_eval.md`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def collection_visited(successes: list[bool], sub_env: int, group_size: int) -> list[int]:
    """Indices one collection sub-env visited, given its per-round success pattern.

    One venv is built per task and reused across rounds, so the reset counter
    persists: each round consumes ``sub_env + group_size*c`` and then advances
    ``c`` by 1, plus 2 more if that round terminated (i.e. succeeded).

    **This model is known to be wrong in both directions (2026-07-29). Treat the
    output as an estimate, not as the set actually visited.**

    * It can UNDERCOUNT. After the autoreset the slot runs a fresh episode and is
      still stepped -- ``rollout.collect`` keeps going until every slot is done --
      so it can terminate again, and each extra termination costs two more.
      Nothing records that: ``just_done`` is masked by ``~done``, so a second
      success is invisible in the episode log this function reads.
    * It can OVERCOUNT. The ``+2`` assumes an autoreset follows, but the vector
      env only autoresets on a subsequent ``venv.step``. If the termination lands
      on the step that makes every slot done, the loop breaks and that reset never
      happens, costing one rather than two.
    * ``succeeded`` is read from the success bit, but ``terminated = done or
      is_success``: the underlying env's ``done`` is a second cause the log cannot
      distinguish.

    The consequence is not a bias in one direction that could be corrected with a
    fudge factor. Later rounds start wherever the counter happens to be, so an
    error in round 0 relocates every subsequent index. The honest reading is that
    the true visited set is unknown without per-reset traces.
    """
    c = 0
    out = []
    for succeeded in successes:
        out.append(sub_env + group_size * c)
        c += 1 + 2 * int(succeeded)
    return out


def grpo_visited(group_size: int) -> set[int]:
    """GRPO builds a FRESH venv per task per iteration, so only the explicit
    reset fires and every iteration starts at the same block."""
    return set(range(group_size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, type=Path,
                    help="collect episodes.jsonl (needs task_key, round, group_index, success)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=15,
                    help="episodes per task the evaluation will run")
    ap.add_argument("--n-init", type=int, default=None,
                    help="len(init_states) per task; if given, windows are checked mod N")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.episodes.read_text().splitlines() if l.strip()]
    by_task: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    for r in rows:
        by_task[r["task_key"]][(int(r["group_index"]), int(r["round"]))] = bool(r["success"])

    grpo = grpo_visited(args.group_size)
    worst = None
    any_impossible = False   # one None anywhere kills the offset route entirely
    for task, cells in sorted(by_task.items()):
        visited: set[int] = set()
        for g in range(args.group_size):
            pattern = [cells.get((g, r), False) for r in range(args.rounds)]
            visited.update(collection_visited(pattern, g, args.group_size))
        blocked = visited | grpo
        # smallest offset whose window avoids every blocked index
        choice = None
        for off in range(0, 4096):
            window = {(off + i) % args.n_init if args.n_init else off + i
                      for i in range(args.n_episodes)}
            if not (window & blocked):
                choice = off
                break
        print(f"{task}: collection visited {sorted(visited)}")
        print(f"    blocked (collection | GRPO) max={max(blocked)}  "
              f"safe offset for {args.n_episodes} eps: {choice}")
        if choice is None:
            print("    !! no disjoint window exists -- an offset cannot make this task held-out")
        # This used to collapse `worst` to None to signal impossibility. If the
        # FIRST task was impossible, `worst` became None and the next task's real
        # offset simply overwrote it -- the impossibility silently vanished. A
        # diagnostic whose whole purpose is to close off the offset route could
        # therefore end up recommending it, so use an explicit flag instead.
        if choice is None:
            any_impossible = True
        elif not any_impossible:
            worst = choice if worst is None else max(worst, choice)

    print()
    if any_impossible or worst is None:
        raise SystemExit("no offset works for every task; do NOT run the offset protocol blind")
    print(f"USE --init_state_offset {worst}   (largest safe offset across tasks)")
    if args.n_init:
        print(f"    checked modulo N={args.n_init}")
    else:
        print("    NOTE: pass --n-init to check wraparound; without it this assumes N is large "
              "enough that (offset + n_episodes) does not wrap.")


if __name__ == "__main__":
    main()
