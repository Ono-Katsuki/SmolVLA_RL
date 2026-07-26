#!/usr/bin/env bash
# (1) baseline + (2) SFT: run lerobot-train on LIBERO-Plus (camera-viewpoint-excluded split)
#
# Usage:
#   bash scripts/train_sft.sh <DRIVE_ROOT> [STEPS] [BATCH_SIZE]
#   SPLIT_MODE=all bash scripts/train_sft.sh <DRIVE_ROOT> 20 1  # wiring check only
#
#   DRIVE_ROOT: persistent storage root on Drive (e.g. /content/drive/MyDrive/SmolVLA_RL)
set -euxo pipefail

DRIVE_ROOT="${1:-/content/drive/MyDrive/SmolVLA_RL}"
STEPS="${2:-100000}"
BATCH_SIZE="${3:-32}"
HF_USER="${HF_USER:-onokatsuki}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
SPLIT_MODE="${SPLIT_MODE:-heldout_safe}"
SMOKE_EPISODES="${SMOKE_EPISODES:-[0]}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
RUN_NAME="${RUN_NAME:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
# upstream (lerobot/smolvla_libero_plus) is a finetune from smolvla_base.
# Fresh init with only VLM weights (empty POLICY_PATH=) yields no success at
# 20k steps (measured 0/30 vs. 33% for the base-initialized reference).
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

SPLIT_FILE="${DRIVE_ROOT}/splits/train_episodes.json"

# Writing checkpoints directly to Drive consumes several GB per save_freq,
# and Colab's rm sends files to Drive's trash without freeing quota. By
# default, write to the VM-local disk during training and copy only the
# latest checkpoint to Drive at the end. The old behavior (direct Drive
# writes, robust to session loss) is available via CKPT_ON_DRIVE=1.
DRIVE_OUTPUT_DIR="${DRIVE_ROOT}/outputs${RUN_NAME:+/${RUN_NAME}}"
CKPT_ON_DRIVE="${CKPT_ON_DRIVE:-0}"
if [ "${CKPT_ON_DRIVE}" = "1" ]; then
    OUTPUT_DIR="${DRIVE_OUTPUT_DIR}"
else
    OUTPUT_DIR="${LOCAL_OUTPUT_ROOT:-/content/outputs}${RUN_NAME:+/${RUN_NAME}}"
fi
CKPT_DIR="${OUTPUT_DIR}/checkpoints/last"
LOG_DIR="${DRIVE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME:-sft}.log"

# lerobot-train refuses to start fresh if OUTPUT_DIR already exists.
# tee would create that directory first as well, so logs go in a sibling.
mkdir -p "$(dirname "${OUTPUT_DIR}")" "${LOG_DIR}"

# Resume check: resume if last/pretrained_model exists.
# If the VM vanished under VM-local operation, restore manually from the Drive copy:
#   cp -r ${DRIVE_OUTPUT_DIR}/checkpoints/<step> ${OUTPUT_DIR}/checkpoints/ and rerun
RESUME_FLAG=""
if [ -d "${CKPT_DIR}/pretrained_model" ]; then
    RESUME_FLAG="--resume=true"
    echo "[train_sft] resume from ${CKPT_DIR}"
fi

# The public LeRobot metadata does not retain perturbation categories, so "all" is smoke-only.
DATASET_ARGS=()
if [ "${SPLIT_MODE}" = "heldout_safe" ]; then
    if [ ! -f "${SPLIT_FILE}" ]; then
        echo "ERROR: ${SPLIT_FILE} is missing. Prepare the leakage-safe split first"
        exit 1
    fi
    DATASET_ARGS+=(--dataset.episodes="$(cat "${SPLIT_FILE}")")
elif [ "${SPLIT_MODE}" = "smoke" ]; then
    DATASET_ARGS+=(--dataset.episodes="${SMOKE_EPISODES}")
elif [ "${SPLIT_MODE}" != "all" ]; then
    echo "ERROR: SPLIT_MODE must be heldout_safe, smoke, or all"
    exit 1
fi
if [ -n "${DATASET_ROOT}" ]; then
    DATASET_ARGS+=(--dataset.root="${DATASET_ROOT}")
fi

POLICY_ARGS=()
if [ -n "${POLICY_PATH}" ]; then
    # Align the dataset's feature names (front/wrist) to the base checkpoint's
    # (camera1/camera2). Same rename_map as the upstream train_config.json.
    POLICY_ARGS+=(--policy.path="${POLICY_PATH}")
    POLICY_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}')
else
    POLICY_ARGS+=(--policy.type=smolvla --policy.load_vlm_weights="${LOAD_VLM_WEIGHTS}")
fi

# TRAIN_VLM=1: train the VLM's LM layers in addition to the action expert ("light brain tuning").
# The vision encoder (SigLIP) stays frozen — visual robustness relies on the pretrained invariances.
# The lr is also lowered to avoid overcooking the LM (override with VLM_LR).
if [ "${TRAIN_VLM:-0}" = "1" ]; then
    POLICY_ARGS+=(--policy.train_expert_only=false)
    POLICY_ARGS+=(--policy.freeze_vision_encoder=true)
    POLICY_ARGS+=(--policy.optimizer_lr="${VLM_LR:-2.5e-5}")
fi

# Do not pass --env.*: with env_eval_freq (default 20k) active, lerobot-train
# builds vec envs for every LIBERO-Plus task up front (1300+ for
# libero_spatial alone), devouring the Colab VM's RAM until it is OOM-killed.
# Evaluation is done by scripts/eval.sh instead.
lerobot-train \
    "${POLICY_ARGS[@]}" \
    --policy.repo_id=${HF_USER}/smolvla_libero_plus_heldout_cam \
    --policy.push_to_hub="${PUSH_TO_HUB}" \
    --dataset.repo_id=lerobot/libero_plus \
    "${DATASET_ARGS[@]}" \
    --output_dir="${OUTPUT_DIR}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --wandb.enable="${WANDB_ENABLE}" \
    ${RESUME_FLAG} \
    2>&1 | tee -a "${LOG_FILE}"

# Sync only the latest checkpoint to Drive (local operation only).
# Older Drive copies are deleted before overwriting, but the trash still
# holds quota, so empty https://drive.google.com/drive/trash periodically.
if [ "${CKPT_ON_DRIVE}" != "1" ]; then
    LATEST="$(readlink -f "${CKPT_DIR}")"
    STEP_NAME="$(basename "${LATEST}")"
    DEST="${DRIVE_OUTPUT_DIR}/checkpoints"
    mkdir -p "${DEST}"
    rm -rf "${DEST:?}/${STEP_NAME}" "${DEST}/last"
    cp -r "${LATEST}" "${DEST}/${STEP_NAME}"
    ln -sfn "${DEST}/${STEP_NAME}" "${DEST}/last" || cp -r "${LATEST}" "${DEST}/last"
    echo "[train_sft] checkpoint synced to ${DEST}/${STEP_NAME}"
fi
