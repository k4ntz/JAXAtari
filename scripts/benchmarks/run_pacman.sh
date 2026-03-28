#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_JAX_PIP_SPEC="jax[cuda13]"
else
  DEFAULT_JAX_PIP_SPEC="jax"
fi
JAX_PIP_SPEC="${JAX_PIP_SPEC:-${DEFAULT_JAX_PIP_SPEC}}"
WANDB_MODE="${WANDB_MODE:-offline}"
ENTITY="${ENTITY:-}"
PROJECT="${PROJECT:-jaxatari-pacman-report}"
SAVE_PATH="${SAVE_PATH:-}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-1}"
VIDEO_STEPS="${VIDEO_STEPS:-}"
RECORD_FINAL_VIDEO="${RECORD_FINAL_VIDEO:-1}"
VIDEO_MAX_STEPS_OBJECT="${VIDEO_MAX_STEPS_OBJECT:-3000}"
VIDEO_MAX_STEPS_PIXEL="${VIDEO_MAX_STEPS_PIXEL:-1000}"
PACMAN_MODS="${PACMAN_MODS:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/report_outputs/pacman_report_$(date +%Y%m%d_%H%M%S)}"

WITH_PPO=0
# QUICK mode selection:
#   1  -> quick profile
#   0  -> full profile
#  -1  -> use config values as-is (default)
QUICK=-1
SKIP_PREFLIGHT=0
SKIP_TRAIN=0
SKIP_VIDEO=0
SHOW_SYSTEM_INFO=1
VIDEO_MODEL="pixel"
AUTO_INSTALL_DEPS=1
PPO_NUM_ENVS_OVERRIDE=""
PPO_NUM_STEPS_OVERRIDE=""
PPO_TOTAL_TIMESTEPS_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmarks/run_pacman.sh [options]

Options:
  --quick                 Fast demo profile (overrides core train hyperparameters).
  --full                  Larger report profile (overrides core train hyperparameters).
                          If neither is set, config values are used as-is (default).
  --with-ppo              Also run PPO training (optional section in report).
  --video-model MODE      MODE in {pixel,object,both}. Default: pixel.
  --ppo-num-envs N        Override PPO NUM_ENVS for object run.
  --ppo-num-steps N       Override PPO NUM_STEPS for object run.
  --ppo-total-steps N     Override PPO TOTAL_TIMESTEPS for object run.
  --mods a,b,c            Override mod list used for per-mod evaluation videos.
  --video-steps N         Override both object/pixel video steps with N.
  --no-final-video        Disable final video recording/evaluation stage.
  --output-dir PATH       Report artifact root directory.
  --skip-preflight        Skip dependency checks.
  --skip-train            Skip training and only run video/eval collection.
  --skip-video            Skip video generation and only run training.
  --no-auto-install       Do not auto-install missing Python deps during preflight.
  --no-system-info        Skip hardware info print.
  -h, --help              Show this help.

Environment overrides:
  PYTHON_BIN, WANDB_MODE, ENTITY, PROJECT, SAVE_PATH, SEED, NUM_SEEDS, OUT_DIR, JAX_PIP_SPEC
  Note: if SAVE_PATH is not set, checkpoints are written to <OUT_DIR>/models.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --full)
      QUICK=0
      shift
      ;;
    --with-ppo)
      WITH_PPO=1
      shift
      ;;
    --video-model)
      VIDEO_MODEL="$2"
      shift 2
      ;;
    --ppo-num-envs)
      PPO_NUM_ENVS_OVERRIDE="$2"
      shift 2
      ;;
    --ppo-num-steps)
      PPO_NUM_STEPS_OVERRIDE="$2"
      shift 2
      ;;
    --ppo-total-steps)
      PPO_TOTAL_TIMESTEPS_OVERRIDE="$2"
      shift 2
      ;;
    --mods)
      PACMAN_MODS="$2"
      shift 2
      ;;
    --video-steps)
      VIDEO_STEPS="$2"
      shift 2
      ;;
    --no-final-video)
      RECORD_FINAL_VIDEO=0
      shift
      ;;
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-video)
      SKIP_VIDEO=1
      shift
      ;;
    --no-auto-install)
      AUTO_INSTALL_DEPS=0
      shift
      ;;
    --no-system-info)
      SHOW_SYSTEM_INFO=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${VIDEO_STEPS}" ]]; then
  VIDEO_MAX_STEPS_OBJECT="${VIDEO_STEPS}"
  VIDEO_MAX_STEPS_PIXEL="${VIDEO_STEPS}"
fi

case "${VIDEO_MODEL}" in
  pixel|object|both) ;;
  *)
    echo "Invalid --video-model '${VIDEO_MODEL}'. Use pixel|object|both." >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"/{logs,metrics,graphs,videos,meta}
if [[ -z "${SAVE_PATH}" ]]; then
  SAVE_PATH="${OUT_DIR}/models"
fi
mkdir -p "${SAVE_PATH}"

OC_TOTAL_TIMESTEPS="${OC_TOTAL_TIMESTEPS:-}"
PIX_TOTAL_TIMESTEPS="${PIX_TOTAL_TIMESTEPS:-}"
PPO_TOTAL_TIMESTEPS="${PPO_TOTAL_TIMESTEPS:-}"
OC_NUM_ENVS="${OC_NUM_ENVS:-}"
PIX_NUM_ENVS="${PIX_NUM_ENVS:-}"
PPO_NUM_ENVS="${PPO_NUM_ENVS:-}"
OC_NUM_STEPS="${OC_NUM_STEPS:-}"
PIX_NUM_STEPS="${PIX_NUM_STEPS:-}"
PPO_NUM_STEPS="${PPO_NUM_STEPS:-}"
PROFILE_MODE="config"

if (( QUICK == 1 )); then
  PROFILE_MODE="quick"
  OC_TOTAL_TIMESTEPS="${OC_TOTAL_TIMESTEPS:-100000}"
  PIX_TOTAL_TIMESTEPS="${PIX_TOTAL_TIMESTEPS:-100000}"
  PPO_TOTAL_TIMESTEPS="${PPO_TOTAL_TIMESTEPS:-100000}"
  OC_NUM_ENVS="${OC_NUM_ENVS:-4}"
  PIX_NUM_ENVS="${PIX_NUM_ENVS:-4}"
  PPO_NUM_ENVS="${PPO_NUM_ENVS:-4}"
  OC_NUM_STEPS="${OC_NUM_STEPS:-32}"
  PIX_NUM_STEPS="${PIX_NUM_STEPS:-16}"
  PPO_NUM_STEPS="${PPO_NUM_STEPS:-16}"
elif (( QUICK == 0 )); then
  PROFILE_MODE="full"
  # Override via env for your actual hardware budget.
  OC_TOTAL_TIMESTEPS="${OC_TOTAL_TIMESTEPS:-200000000}"
  PIX_TOTAL_TIMESTEPS="${PIX_TOTAL_TIMESTEPS:-100000000}"
  PPO_TOTAL_TIMESTEPS="${PPO_TOTAL_TIMESTEPS:-50000000}"
  OC_NUM_ENVS="${OC_NUM_ENVS:-8192}"
  PIX_NUM_ENVS="${PIX_NUM_ENVS:-128}"
  PPO_NUM_ENVS="${PPO_NUM_ENVS:-8}"
  OC_NUM_STEPS="${OC_NUM_STEPS:-32}"
  PIX_NUM_STEPS="${PIX_NUM_STEPS:-4}"
  PPO_NUM_STEPS="${PPO_NUM_STEPS:-128}"
fi

if [[ -n "${PPO_NUM_ENVS_OVERRIDE}" ]]; then
  PPO_NUM_ENVS="${PPO_NUM_ENVS_OVERRIDE}"
fi
if [[ -n "${PPO_NUM_STEPS_OVERRIDE}" ]]; then
  PPO_NUM_STEPS="${PPO_NUM_STEPS_OVERRIDE}"
fi
if [[ -n "${PPO_TOTAL_TIMESTEPS_OVERRIDE}" ]]; then
  PPO_TOTAL_TIMESTEPS="${PPO_TOTAL_TIMESTEPS_OVERRIDE}"
fi

WANDB_ROOT="${WANDB_DIR:-${OUT_DIR}/wandb}"
mkdir -p "${WANDB_ROOT}"
export WANDB_DIR="${WANDB_ROOT}"

log_note() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

print_system_info() {
  echo "=== Report Hardware Info ==="
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "OS: ${PRETTY_NAME:-unknown}"
  elif command -v sw_vers >/dev/null 2>&1; then
    echo "OS: $(sw_vers -productName) $(sw_vers -productVersion)"
  else
    echo "OS: $(uname -srm)"
  fi

  if command -v lscpu >/dev/null 2>&1; then
    echo "CPU: $(lscpu | awk -F: '/Model name/{print $2; exit}' | xargs)"
    echo "CPU cores: $(lscpu | awk -F: '/^CPU\\(s\\)/{print $2; exit}' | xargs)"
  elif command -v sysctl >/dev/null 2>&1; then
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  fi

  if command -v free >/dev/null 2>&1; then
    echo "RAM: $(free -h | awk '/^Mem:/{print $2}')"
  elif command -v sysctl >/dev/null 2>&1; then
    ram_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ -n "${ram_bytes}" ]]; then
      echo "RAM: $(awk -v b="${ram_bytes}" 'BEGIN{printf "%.1f GB", b/1024/1024/1024}')"
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s):"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  - /'
  else
    echo "GPU(s): nvidia-smi not found (CPU-only or non-NVIDIA setup)."
  fi

  "${PYTHON_BIN}" - <<'PY'
import sys
print(f"Python Version: {sys.version.split()[0]}")
try:
    import jax
    print(f"JAX Version: {jax.__version__}")
    try:
        import jaxlib
        print(f"jaxlib Version: {jaxlib.__version__}")
    except Exception:
        print("jaxlib Version: not detected")
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
except Exception as exc:
    print(f"JAX check failed: {exc}")
PY
  echo "============================"
}

check_modules() {
  local specs="$1"
  PY_REQ_SPECS="${specs}" AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS}" PYTHON_BIN_HINT="${PYTHON_BIN}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import importlib.util
import subprocess

missing = []
for item in [x.strip() for x in os.environ.get("PY_REQ_SPECS", "").split(",") if x.strip()]:
    if ":" in item:
        mod, pkg = item.split(":", 1)
    else:
        mod, pkg = item, item
    if importlib.util.find_spec(mod) is None:
        missing.append((mod, pkg))

if missing:
    py = sys.executable
    auto_install = os.environ.get("AUTO_INSTALL_DEPS", "1") == "1"
    print("Missing required Python modules:")
    for mod, pkg in missing:
        print(f"  - module '{mod}' (install package: {pkg})")

    pkgs = sorted({pkg for _, pkg in missing})
    if auto_install:
        print("\nAuto-installing missing packages...")
        cmd = [py, "-m", "pip", "install", *pkgs]
        print("Running:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"Auto-install failed with exit code {exc.returncode}")
            sys.exit(2)

        still_missing = []
        for mod, pkg in missing:
            if importlib.util.find_spec(mod) is None:
                still_missing.append((mod, pkg))
        if still_missing:
            print("\nSome modules are still missing after auto-install:")
            for mod, pkg in still_missing:
                print(f"  - module '{mod}' (package: {pkg})")
            sys.exit(2)
        print("Auto-install successful.")
    else:
        print("\nAuto-install is disabled. Install manually with:")
        print(f"  {py} -m pip install {' '.join(pkgs)}")
        sys.exit(2)
PY
}

discover_mods() {
  "${PYTHON_BIN}" - <<'PY'
from jaxatari.games.mods.pacman_mods import PacmanEnvMod
print(",".join(PacmanEnvMod.REGISTRY.keys()))
PY
}

if (( SHOW_SYSTEM_INFO == 1 )); then
  print_system_info | tee "${OUT_DIR}/meta/hardware_info.txt"
fi

if (( SKIP_PREFLIGHT == 0 )); then
  check_modules "jax:${JAX_PIP_SPEC},numpy:numpy,absl:absl-py,ale_py:ale-py,gymnasium:gymnasium,gymnax:gymnax,ml_dtypes:ml-dtypes,scipy:scipy,toolz:toolz,chex:chex,flax:flax,optax:optax,distrax:distrax,hydra:hydra-core,omegaconf:omegaconf,wandb:wandb,safetensors:safetensors,matplotlib:matplotlib,pandas:pandas,tqdm:tqdm,pygame:pygame,cv2:opencv-python-headless,psutil:psutil,ocatari:ocatari,imageio:imageio,PIL:pillow,jinja2:jinja2"
fi

if (( SKIP_VIDEO == 0 )); then
  if [[ -z "${PACMAN_MODS}" ]]; then
    PACMAN_MODS="$(discover_mods)"
  fi
else
  PACMAN_MODS="${PACMAN_MODS:-}"
fi

IFS=',' read -r -a MOD_LIST <<< "${PACMAN_MODS}"

run_and_log() {
  local log_file="$1"
  shift
  log_note "Running: $*"
  "$@" 2>&1 | tee "${log_file}"
}

run_pqn_train() {
  local label="$1"
  local alg_cfg="$2"
  local total_timesteps="$3"
  local num_envs="$4"
  local num_steps="$5"

  local log_file="${OUT_DIR}/logs/train_${label}.log"
  local cmd=(
    "${PYTHON_BIN}" "scripts/benchmarks/pqn_agent.py" "+alg=${alg_cfg}"
    "WANDB_MODE=${WANDB_MODE}"
    "PROJECT=${PROJECT}"
    "SAVE_PATH=${SAVE_PATH}"
    "SEED=${SEED}"
    "NUM_SEEDS=${NUM_SEEDS}"
    "EXPORT_METRICS=True"
    "REPORT_OUTPUT_DIR=${OUT_DIR}"
    "NAME=${label}"
    "REPORT_TAG=${label}"
  )
  if [[ -n "${total_timesteps}" ]]; then
    cmd+=("alg.TOTAL_TIMESTEPS=${total_timesteps}")
    cmd+=("alg.TOTAL_TIMESTEPS_DECAY=${total_timesteps}")
  fi
  if [[ -n "${num_envs}" ]]; then
    cmd+=("alg.NUM_ENVS=${num_envs}")
  fi
  if [[ -n "${num_steps}" ]]; then
    cmd+=("alg.NUM_STEPS=${num_steps}")
  fi
  if [[ -n "${ENTITY}" ]]; then
    cmd+=("ENTITY=${ENTITY}")
  fi
  run_and_log "${log_file}" "${cmd[@]}"
}

run_ppo_train() {
  local label="$1"
  local alg_cfg="$2"
  local total_timesteps="$3"
  local num_envs="$4"
  local num_steps="$5"

  local log_file="${OUT_DIR}/logs/train_${label}.log"
  local cmd=(
    "${PYTHON_BIN}" "scripts/benchmarks/ppo_agent.py" "+alg=${alg_cfg}"
    "WANDB_MODE=${WANDB_MODE}"
    "PROJECT=${PROJECT}"
    "SAVE_PATH=${SAVE_PATH}"
    "SEED=${SEED}"
    "NUM_SEEDS=${NUM_SEEDS}"
    "EXPORT_METRICS=True"
    "REPORT_OUTPUT_DIR=${OUT_DIR}"
    "NAME=${label}"
  )
  if [[ -n "${total_timesteps}" ]]; then
    cmd+=("alg.TOTAL_TIMESTEPS=${total_timesteps}")
    cmd+=("alg.TOTAL_TIMESTEPS_DECAY=${total_timesteps}")
  fi
  if [[ -n "${num_envs}" ]]; then
    cmd+=("alg.NUM_ENVS=${num_envs}")
  fi
  if [[ -n "${num_steps}" ]]; then
    cmd+=("alg.NUM_STEPS=${num_steps}")
  fi
  if [[ -n "${ENTITY}" ]]; then
    cmd+=("ENTITY=${ENTITY}")
  fi
  run_and_log "${log_file}" "${cmd[@]}"
}

collect_new_videos() {
  local marker="$1"
  local dest_prefix="$2"
  local idx=0
  while IFS= read -r -d '' vid; do
    idx=$((idx + 1))
    base="$(basename "${vid}")"
    cp -f "${vid}" "${OUT_DIR}/videos/${dest_prefix}_${idx}_${base}"
  done < <(find "${WANDB_ROOT}" -type f -path "*/media/videos/*.mp4" -newer "${marker}" -print0 2>/dev/null)
}

snapshot_configs() {
  local cfg_dir="${OUT_DIR}/meta/config_snapshots"
  mkdir -p "${cfg_dir}/alg"
  cp -f "scripts/benchmarks/config/config.yaml" "${cfg_dir}/config.yaml"
  cp -f "scripts/benchmarks/config/alg/pqn_pacman_object.yaml" "${cfg_dir}/alg/pqn_pacman_object.yaml"
  cp -f "scripts/benchmarks/config/alg/pqn_pacman_pixel.yaml" "${cfg_dir}/alg/pqn_pacman_pixel.yaml"
  cp -f "scripts/benchmarks/config/alg/ppo_pacman_object.yaml" "${cfg_dir}/alg/ppo_pacman_object.yaml"
}

generate_report_markdown() {
  local md_script="scripts/benchmarks/report_sum.py"
  if [[ ! -f "${md_script}" ]]; then
    log_note "Markdown generator script not found: ${md_script}"
    return
  fi
  "${PYTHON_BIN}" "${md_script}" \
    --out-dir "${OUT_DIR}" \
    --mods "${PACMAN_MODS}" \
    --with-ppo "${WITH_PPO}" \
    --video-model "${VIDEO_MODEL}" \
    --quick "${QUICK}" \
    > "${OUT_DIR}/logs/generate_report_markdown.log" 2>&1 || {
      log_note "Failed to generate markdown draft. Check logs/generate_report_markdown.log"
      return
    }
}

run_pqn_video_eval_once() {
  local model_cfg="$1"      # pqn_pacman_pixel | pqn_pacman_object
  local model_label="$2"    # pixel | object
  local mod_label="$3"      # base or mod key
  local video_steps="$4"
  local eval_mods_override="[]"
  if [[ "${mod_label}" != "base" ]]; then
    eval_mods_override="[${mod_label}]"
  fi

  local marker="${OUT_DIR}/meta/.video_marker_pqn_${model_label}_${mod_label}_$$"
  touch "${marker}"

  local log_file="${OUT_DIR}/logs/video_pqn_${model_label}_${mod_label}.log"
  local cmd=(
    "${PYTHON_BIN}" "scripts/benchmarks/pqn_test.py" "+alg=${model_cfg}"
    "WANDB_MODE=${WANDB_MODE}"
    "PROJECT=${PROJECT}"
    "SAVE_PATH=${SAVE_PATH}"
    "SEED=${SEED}"
    "NUM_SEEDS=${NUM_SEEDS}"
    "alg.ENV_NAME=Pacman"
    "alg.TRAIN_MODS=null"
    "alg.EVAL_MODS=${eval_mods_override}"
    "alg.MOD_NAME=null"
    "alg.TEST_NUM_ENVS=1"
    "alg.TEST_NUM_STEPS=${video_steps}"
    "alg.RECORD_VIDEO=True"
    "NAME=video_pqn_${model_label}_${mod_label}"
  )
  if [[ -n "${ENTITY}" ]]; then
    cmd+=("ENTITY=${ENTITY}")
  fi

  run_and_log "${log_file}" "${cmd[@]}"
  collect_new_videos "${marker}" "pqn_${model_label}_${mod_label}"
  rm -f "${marker}"
}

run_ppo_video_eval_once() {
  local model_cfg="$1"      # ppo_pacman_object
  local model_label="$2"    # object
  local mod_label="$3"      # base or mod key
  local video_steps="$4"
  local eval_mods_override="[]"
  if [[ "${mod_label}" != "base" ]]; then
    eval_mods_override="[${mod_label}]"
  fi

  local marker="${OUT_DIR}/meta/.video_marker_ppo_${model_label}_${mod_label}_$$"
  touch "${marker}"

  local log_file="${OUT_DIR}/logs/video_ppo_${model_label}_${mod_label}.log"
  local cmd=(
    "${PYTHON_BIN}" "scripts/benchmarks/ppo_test.py" "+alg=${model_cfg}"
    "WANDB_MODE=${WANDB_MODE}"
    "PROJECT=${PROJECT}"
    "SAVE_PATH=${SAVE_PATH}"
    "SEED=${SEED}"
    "NUM_SEEDS=${NUM_SEEDS}"
    "alg.ENV_NAME=Pacman"
    "alg.TRAIN_MODS=null"
    "alg.EVAL_MODS=${eval_mods_override}"
    "alg.MOD_NAME=null"
    "alg.TEST_NUM_ENVS=1"
    "alg.TEST_NUM_STEPS=${video_steps}"
    "alg.RECORD_VIDEO=True"
    "NAME=video_ppo_${model_label}_${mod_label}"
  )
  if [[ -n "${ENTITY}" ]]; then
    cmd+=("ENTITY=${ENTITY}")
  fi

  run_and_log "${log_file}" "${cmd[@]}"
  collect_new_videos "${marker}" "ppo_${model_label}_${mod_label}"
  rm -f "${marker}"
}

if (( SKIP_TRAIN == 0 )); then
  log_note "Training PQN object-centric..."
  run_pqn_train "pqn_object_report" "pqn_pacman_object" "${OC_TOTAL_TIMESTEPS}" "${OC_NUM_ENVS}" "${OC_NUM_STEPS}"

  log_note "Training PQN pixel..."
  run_pqn_train "pqn_pixel_report" "pqn_pacman_pixel" "${PIX_TOTAL_TIMESTEPS}" "${PIX_NUM_ENVS}" "${PIX_NUM_STEPS}"

  if (( WITH_PPO == 1 )); then
    log_note "Training PPO object-centric (optional)..."
    run_ppo_train "ppo_object_report" "ppo_pacman_object" "${PPO_TOTAL_TIMESTEPS}" "${PPO_NUM_ENVS}" "${PPO_NUM_STEPS}"
  fi
fi

if (( SKIP_VIDEO == 0 )) && (( RECORD_FINAL_VIDEO == 1 )); then
  run_video_block_pqn() {
    local model_cfg="$1"
    local model_label="$2"
    local video_steps="$3"
    log_note "Generating PQN base video for ${model_label} model..."
    run_pqn_video_eval_once "${model_cfg}" "${model_label}" "base" "${video_steps}"
    for mod in "${MOD_LIST[@]}"; do
      [[ -z "${mod}" ]] && continue
      log_note "Generating PQN ${model_label} video for mod: ${mod}"
      run_pqn_video_eval_once "${model_cfg}" "${model_label}" "${mod}" "${video_steps}"
    done
  }

  run_video_block_ppo() {
    local model_cfg="$1"
    local model_label="$2"
    local video_steps="$3"
    log_note "Generating PPO base video for ${model_label} model..."
    run_ppo_video_eval_once "${model_cfg}" "${model_label}" "base" "${video_steps}"
    for mod in "${MOD_LIST[@]}"; do
      [[ -z "${mod}" ]] && continue
      log_note "Generating PPO ${model_label} video for mod: ${mod}"
      run_ppo_video_eval_once "${model_cfg}" "${model_label}" "${mod}" "${video_steps}"
    done
  }

  case "${VIDEO_MODEL}" in
    pixel)
      run_video_block_pqn "pqn_pacman_pixel" "pixel" "${VIDEO_MAX_STEPS_PIXEL}"
      if (( WITH_PPO == 1 )); then
        log_note "Skipping PPO pixel videos: PPO runs object-centric only."
      fi
      ;;
    object)
      run_video_block_pqn "pqn_pacman_object" "object" "${VIDEO_MAX_STEPS_OBJECT}"
      if (( WITH_PPO == 1 )); then
        run_video_block_ppo "ppo_pacman_object" "object" "${VIDEO_MAX_STEPS_OBJECT}"
      fi
      ;;
    both)
      run_video_block_pqn "pqn_pacman_pixel" "pixel" "${VIDEO_MAX_STEPS_PIXEL}"
      run_video_block_pqn "pqn_pacman_object" "object" "${VIDEO_MAX_STEPS_OBJECT}"
      if (( WITH_PPO == 1 )); then
        run_video_block_ppo "ppo_pacman_object" "object" "${VIDEO_MAX_STEPS_OBJECT}"
      fi
      ;;
  esac
elif (( SKIP_VIDEO == 0 )); then
  log_note "Skipping final video generation because RECORD_FINAL_VIDEO=0."
fi

{
  echo "Report output directory: ${OUT_DIR}"
  echo "Models directory: ${SAVE_PATH}"
  echo "WANDB_MODE: ${WANDB_MODE}"
  echo "SEED: ${SEED}"
  echo "NUM_SEEDS: ${NUM_SEEDS}"
  echo "Mods evaluated: ${PACMAN_MODS}"
  echo "Video model mode: ${VIDEO_MODEL}"
  echo "RECORD_FINAL_VIDEO: ${RECORD_FINAL_VIDEO}"
  echo "VIDEO_MAX_STEPS_OBJECT: ${VIDEO_MAX_STEPS_OBJECT}"
  echo "VIDEO_MAX_STEPS_PIXEL: ${VIDEO_MAX_STEPS_PIXEL}"
  echo "Run profile: ${PROFILE_MODE}"
  echo "With PPO: ${WITH_PPO}"
  echo "PPO mode: object_only"
  echo "JAX auto-install package: ${JAX_PIP_SPEC}"
  echo "WANDB_DIR: ${WANDB_ROOT}"
  echo "OC_TOTAL_TIMESTEPS: ${OC_TOTAL_TIMESTEPS}"
  echo "PIX_TOTAL_TIMESTEPS: ${PIX_TOTAL_TIMESTEPS}"
  echo "PPO_TOTAL_TIMESTEPS: ${PPO_TOTAL_TIMESTEPS}"
  echo "OC_NUM_ENVS: ${OC_NUM_ENVS}"
  echo "PIX_NUM_ENVS: ${PIX_NUM_ENVS}"
  echo "PPO_NUM_ENVS: ${PPO_NUM_ENVS}"
  echo "OC_NUM_STEPS: ${OC_NUM_STEPS}"
  echo "PIX_NUM_STEPS: ${PIX_NUM_STEPS}"
  echo "PPO_NUM_STEPS: ${PPO_NUM_STEPS}"
} > "${OUT_DIR}/meta/run_summary.txt"

snapshot_configs
generate_report_markdown

log_note "Done. Artifacts generated under: ${OUT_DIR}"
log_note "Graphs: ${OUT_DIR}/graphs"
log_note "Metrics CSV/NPZ: ${OUT_DIR}/metrics"
log_note "Videos: ${OUT_DIR}/videos"
log_note "Report draft: ${OUT_DIR}/PACMAN_REPORT_DRAFT.md"
