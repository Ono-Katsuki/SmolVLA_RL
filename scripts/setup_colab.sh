#!/usr/bin/env bash
# Colab environment setup: LeRobot + the full LIBERO-Plus stack
set -eux

LIBERO_PLUS_DIR="/content/LIBERO-plus"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_DIR}/libero/libero"

# ---------- 1. LeRobot (SmolVLA extras) ----------
pip install -q -r requirements.txt
# The Colab-preinstalled version can fail to fetch Xet signed URLs, so update it.
pip install -q -U hf-xet
unset HF_HUB_DISABLE_XET
export HF_XET_HIGH_PERFORMANCE=1

# ---------- 2. LIBERO-Plus (provides import libero in place of LIBERO) ----------
pip uninstall -y hf-libero libero 2>/dev/null || true
if [ ! -d "${LIBERO_PLUS_DIR}/.git" ]; then
    git clone https://github.com/sylvestf/LIBERO-plus.git "${LIBERO_PLUS_DIR}"
fi
git -C "${LIBERO_PLUS_DIR}" checkout 4976dc3
pip install -q --no-deps -e "${LIBERO_PLUS_DIR}"

# ---------- 4. MuJoCo / EGL (headless rendering on Colab) ----------
apt-get update -q
apt-get install -y -q libegl1 libgl1 libosmesa6 xvfb unzip \
    libexpat1 libfontconfig1-dev libmagickwand-dev
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ---------- 4. LIBERO-Plus assets (6.4 GB; first time only) ----------
# Can be skipped with SKIP_LIBERO_ASSETS=1 when validating training first.
# Evaluation that launches environments requires the assets.
if [ "${SKIP_LIBERO_ASSETS:-0}" = "1" ]; then
    echo "==== skipping LIBERO-Plus assets (training-only setup) ===="
elif [ ! -d "${LIBERO_PLUS_ROOT}/assets" ]; then
    python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Sylvest/LIBERO-plus",
    repo_type="dataset",
    filename="assets.zip",
    local_dir="/content/libero-plus-assets",
)
PY
    rm -rf /content/libero-plus-assets/extract
    unzip -q /content/libero-plus-assets/assets.zip -d /content/libero-plus-assets/extract
    ASSETS_DIR="$(find /content/libero-plus-assets/extract -type d -name assets | head -1)"
    if [ -z "${ASSETS_DIR}" ]; then
        echo "ERROR: no assets directory inside assets.zip"
        exit 1
    fi
    mv "${ASSETS_DIR}" "${LIBERO_PLUS_ROOT}/assets"
fi

# Keep LIBERO's first import from stalling at input()
mkdir -p "${HOME}/.libero"
printf 'assets: %s\nbddl_files: %s\ndatasets: %s\ninit_states: %s\n' \
    "${LIBERO_PLUS_ROOT}/assets" \
    "${LIBERO_PLUS_ROOT}/bddl_files" \
    "${LIBERO_PLUS_DIR}/libero/datasets" \
    "${LIBERO_PLUS_ROOT}/init_files" \
    > "${HOME}/.libero/config.yaml"

# ---------- 5. sanity check ----------
python -c "import lerobot; print('lerobot', lerobot.__version__)"
PYTHONPATH="${LIBERO_PLUS_DIR}:${PYTHONPATH:-}" python -c "import libero; print('libero-plus', libero.__file__)"
PYTHONPATH="${LIBERO_PLUS_DIR}:${PYTHONPATH:-}" python -c "from libero.libero import benchmark; print('LIBERO-plus suites', sorted(benchmark.get_benchmark_dict()))"
echo "==== setup complete ===="
