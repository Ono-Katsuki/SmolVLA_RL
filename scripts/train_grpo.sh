#!/usr/bin/env bash
# GRPO training (starting from the SFT ckpt; Flow-SDE + PPO-style GRPO)
#
# Usage:
#   bash scripts/train_grpo.sh <DRIVE_ROOT> [SFT_CKPT]
#   CONFIG=configs/grpo_smoke.yaml bash scripts/train_grpo.sh <DRIVE_ROOT> <SFT_CKPT>
#
#   SFT_CKPT: a checkpoints/last-style directory (containing pretrained_model)
set -euxo pipefail

DRIVE_ROOT="${1:-/content/drive/MyDrive/SmolVLA_RL}"
SFT_CKPT="${2:-${DRIVE_ROOT}/outputs/checkpoints/last}"
CONFIG="${CONFIG:-configs/grpo_libero.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_ROOT}/grpo_outputs}"
LOG_DIR="${DRIVE_ROOT}/logs"
mkdir -p "${LOG_DIR}"

python -m src.grpo.train_grpo \
    --config "${CONFIG}" \
    --sft_checkpoint "${SFT_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/grpo.log"
