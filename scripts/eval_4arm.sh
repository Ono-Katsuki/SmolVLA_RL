#!/usr/bin/env bash
# Evaluate the 4 arms (SFT / RS-SFT / DPO / GRPO) on in-dist / held-out.
# Each (method, mode) is dispatched to src.eval_heldout, leaving result.json
# in eval_out/<method>/<mode>/. Finally make_results_table.py emits a comparison table.
#
# Usage (after SFT→collect→RS-SFT→DPO→GRPO are done, on Colab):
#   bash scripts/eval_4arm.sh
# Overridable via env vars (checkpoint paths, task count, episode count):
#   TASKS=8 TRIALS=15 SUITE=libero_spatial bash scripts/eval_4arm.sh
set -euo pipefail

SUITE="${SUITE:-libero_spatial}"
TASKS="${TASKS:-8}"          # tasks_per_suite (drawn separately from in-dist/held-out)
TRIALS="${TRIALS:-15}"       # episodes per task
BATCH="${BATCH:-8}"   # async parallelism. On the A100 VM (12vCPU/83GB), 8 runs safely and evaluation is ~2x faster
SEED="${SEED:-42}"
LIBERO_ROOT="${LIBERO_ROOT:-/content/LIBERO-plus}"
OUT="${OUT:-/content/eval_out}"
ENGINE="${ENGINE:-cli}"   # switchable to inproc (cross-check one task on both engines first)

# method name → checkpoint directory (only existing ones are evaluated)
declare -A CKPT=(
  [sft]="${SFT_CKPT:-/content/outputs/sft_base_heldout/checkpoints/last}"
  [rs_sft]="${RS_SFT_CKPT:-/content/rs_sft_run/checkpoints/last}"
  [dpo]="${DPO_CKPT:-/content/dpo_run/checkpoints/last}"
  [grpo]="${GRPO_CKPT:-/content/grpo_run7/ckpt_latest}"
)

echo "== 4-arm eval: suite=${SUITE} tasks=${TASKS} trials=${TRIALS} =="
for method in sft rs_sft dpo grpo; do
  ck="${CKPT[$method]}"
  if [ ! -e "${ck}" ] && [ ! -e "${ck}/pretrained_model" ]; then
    echo "[skip] ${method}: no checkpoint (${ck})"
    continue
  fi
  for mode in in_dist heldout; do
    od="${OUT}/${method}/${mode}"
    echo "--- eval ${method} / ${mode} -> ${od}"
    python -m src.eval_heldout \
      --checkpoint "${ck}" \
      --suite "${SUITE}" \
      --mode "${mode}" \
      --engine "${ENGINE}" \
      --eval_trials "${TRIALS}" \
      --eval_batch_size "${BATCH}" \
      --tasks_per_suite "${TASKS}" \
      --seed "${SEED}" \
      --libero_root "${LIBERO_ROOT}" \
      --output_dir "${od}"
  done
done

echo "== comparison table =="
python -m src.make_results_table --eval_root "${OUT}" --out "${OUT}/results_table.md"
cat "${OUT}/results_table.md"
