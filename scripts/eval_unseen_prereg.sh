#!/usr/bin/env bash
# Preregistered unseen-task generalization test: GRPO vs its own SFT initialization.
#
# The design is FIXED by data/eval/PREREGISTRATION_unseen_task_eval.md, which was
# committed before this script was ever run. The parameters are hard-coded here on
# purpose: they are not knobs. If you find yourself editing them after seeing a
# result, you are no longer running the preregistered study.
#
#   48 unseen tasks x 3 paired initial states x 2 arms = 288 episodes
#
# Usage (Colab, after cell 2 has cloned the repo and mounted Drive):
#   SFT_CKPT=... GRPO_CKPT=... OUT=... bash scripts/eval_unseen_prereg.sh
set -euo pipefail

SUITE=libero_spatial
MODE=in_dist          # single pool -> a single test, no multiplicity correction
TASKS=48
TRIALS=3
BATCH=8
SEED=20260725
OFFSET=0              # unseen task => every initial state is held out by construction
MIN_INIT=3            # Amendment 1: N is bimodal (1 or 50); N=1 tasks cannot give 3 episodes
ENGINE=inproc         # the cli engine ignores initial-state control

# 8 GRPO training tasks + the 16 tasks of the earlier exploratory probe. Excluded
# before sampling so the preregistered draw cannot depend on results already seen.
EXCLUDE=79,108,1477,1530,1817,1955,2126,2172,1685,228,51,1894,563,501,457,285,935,665,620,748,733,722,679,660

LIBERO_ROOT="${LIBERO_ROOT:-/content/LIBERO-plus}"
OUT="${OUT:-/content/eval_unseen_prereg}"

: "${SFT_CKPT:?set SFT_CKPT to the checkpoint GRPO was initialized from}"
: "${GRPO_CKPT:?set GRPO_CKPT to the run8 final checkpoint}"

for pair in "sft:${SFT_CKPT}" "grpo:${GRPO_CKPT}"; do
  arm="${pair%%:*}"; ck="${pair#*:}"
  if [ ! -e "${ck}" ] && [ ! -e "${ck}/pretrained_model" ]; then
    echo "FATAL: ${arm} checkpoint not found: ${ck}" >&2
    exit 1
  fi
  echo "== ${arm}: ${ck}"
  python -m src.eval_heldout \
    --checkpoint "${ck}" \
    --suite "${SUITE}" --mode "${MODE}" --engine "${ENGINE}" \
    --tasks_per_suite "${TASKS}" --eval_trials "${TRIALS}" \
    --eval_batch_size "${BATCH}" --seed "${SEED}" \
    --init_state_offset "${OFFSET}" --min_init_states "${MIN_INIT}" \
    --exclude_task_ids "${EXCLUDE}" \
    --libero_root "${LIBERO_ROOT}" \
    --output_dir "${OUT}/${arm}"
done

# The pairing is the whole design. If the two arms did not evaluate the same
# tasks, nothing downstream is a paired comparison and the run is void.
python - "${OUT}" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
ids, eps = {}, {}
for arm in ("sft", "grpo"):
    r = json.loads((root / arm / "result.json").read_text())
    ids[arm] = list(r["task_ids"])
    eps[arm] = r["n_total"]
    errored = [t["task_id"] for t in r["per_task"] if t.get("error")]
    if errored:
        print(f"note: {arm} had {len(errored)} errored task(s): {errored}")
if ids["sft"] != ids["grpo"]:
    sys.exit(f"VOID: arms evaluated different task lists\n sft={ids['sft']}\n grpo={ids['grpo']}")
if eps["sft"] != eps["grpo"]:
    sys.exit(
        f"IMBALANCE: sft={eps['sft']} vs grpo={eps['grpo']} episodes. A task errored for "
        "one arm but not the other. The preregistration says to repair and rerun (delete "
        "that arm's task_<id>.json and re-run this script), or to report the imbalance -- "
        "NOT to drop the task silently."
    )
print(f"OK: paired on {len(ids['sft'])} identical tasks, {eps['sft']} episodes per arm")
PY
