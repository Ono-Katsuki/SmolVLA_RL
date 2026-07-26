#!/usr/bin/env bash
# 4 suites × 10 tasks × 10 episodes = 400-episode evaluation
#
# Usage:
#   bash scripts/eval.sh <CKPT_DIR> <MODE>
#     MODE: in_dist | heldout
set -eux

CKPT_DIR="${1:?checkpoint dir required}"
MODE="${2:-in_dist}"
EVAL_TRIALS="${EVAL_TRIALS:-10}"
TASKS_PER_SUITE="${TASKS_PER_SUITE:-10}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

RESULT_ROOT="${CKPT_DIR}/eval_${MODE}"
mkdir -p "${RESULT_ROOT}"

for SUITE in "${SUITES[@]}"; do
    echo "===== eval ${SUITE} (${MODE}) ====="
    python src/eval_heldout.py \
        --checkpoint "${CKPT_DIR}" \
        --suite "${SUITE}" \
        --mode "${MODE}" \
        --eval_trials "${EVAL_TRIALS}" \
        --tasks_per_suite "${TASKS_PER_SUITE}" \
        --output_dir "${RESULT_ROOT}/${SUITE}"
done

# Aggregate
python src/eval_heldout.py --aggregate --output_dir "${RESULT_ROOT}"
