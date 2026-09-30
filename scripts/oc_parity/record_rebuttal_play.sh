#!/usr/bin/env bash
# Record human-play OC trajectories for the rebuttal game set.
#
# Defaults: 100 warmup frames (not saved) + 1600 recorded frames @ 30 FPS play.
# With no arguments, walks through all games sequentially (press Enter between them).
#
# Usage:
#   bash scripts/oc_parity/record_rebuttal_play.sh           # all games, in order
#   bash scripts/oc_parity/record_rebuttal_play.sh freeway   # one game
#   bash scripts/oc_parity/record_rebuttal_play.sh print     # print commands only
#
# Env overrides: CORPUS SKIP_FRAMES STEPS TAG FPS
# After recording:
#   PYTHONPATH=scripts python scripts/oc_parity/run_rebuttal_eval.py \
#     --corpus data/oc_parity_corpus --cpu --n 150 --num-t0 5 --traj-tag play

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
eval "$(pyenv init -)"
pyenv activate cuda_jax

CORPUS="${CORPUS:-data/oc_parity_corpus}"
RAW="${CORPUS}/raw"
SKIP="${SKIP_FRAMES:-100}"
STEPS="${STEPS:-1600}"
TAG="${TAG:-play01}"
FPS="${FPS:-30}"

mkdir -p "$RAW"

# jax_key:OC_Atari_name
GAMES=(
  "freeway:Freeway"
  "montezumarevenge:MontezumaRevenge"
  "pong:Pong"
  "bankheist:BankHeist"
  "seaquest:Seaquest"
  "skiing:Skiing"
)

record_one() {
  local jax_key="$1"
  local oc_name="$2"
  local idx="$3"
  local total="$4"
  local out="${RAW}/${jax_key}_${TAG}.npz"
  echo
  echo "============================================================"
  echo "  [${idx}/${total}] ${jax_key}  (OC=${oc_name})"
  echo "  out: ${out}"
  echo "  warmup=${SKIP}  record=${STEPS}  fps=${FPS}"
  echo "============================================================"
  echo "Play in the window until the step cap (or quit early if supported)."
  read -r -p "Press Enter to start this game (Ctrl-C to abort)... "
  PYTHONPATH=scripts python scripts/oc_parity/record_ocatari.py \
    -g "$oc_name" \
    --mode play \
    --skip-frames "$SKIP" \
    --steps "$STEPS" \
    --fps "$FPS" \
    --out "$out"
  echo "Wrote ${out}"
}

print_all() {
  echo "Manual rebuttal recordings (warmup=${SKIP}, steps=${STEPS}, tag=${TAG}):"
  echo
  for entry in "${GAMES[@]}"; do
    local jax_key="${entry%%:*}"
    local oc_name="${entry##*:}"
    echo "PYTHONPATH=scripts python scripts/oc_parity/record_ocatari.py \\"
    echo "  -g ${oc_name} --mode play --skip-frames ${SKIP} --steps ${STEPS} --fps ${FPS} \\"
    echo "  --out ${RAW}/${jax_key}_${TAG}.npz"
    echo
  done
  echo "Then evaluate:"
  echo "PYTHONPATH=scripts python scripts/oc_parity/run_rebuttal_eval.py \\"
  echo "  --corpus ${CORPUS} --cpu --n 150 --num-t0 5 --traj-tag play"
}

record_all() {
  local total="${#GAMES[@]}"
  local i=0
  echo "Will record ${total} games into ${RAW}/*_${TAG}.npz"
  echo "Order: freeway → montezuma → pong → bankheist → seaquest → skiing"
  for entry in "${GAMES[@]}"; do
    i=$((i + 1))
    record_one "${entry%%:*}" "${entry##*:}" "$i" "$total"
  done
  echo
  echo "All done. Evaluate with:"
  echo "PYTHONPATH=scripts python scripts/oc_parity/run_rebuttal_eval.py \\"
  echo "  --corpus ${CORPUS} --cpu --n 150 --num-t0 5 --traj-tag play"
}

target="${1:-all}"

case "$target" in
  print|--print|-p)
    print_all
    ;;
  all|"")
    record_all
    ;;
  *)
    found=""
    total="${#GAMES[@]}"
    for entry in "${GAMES[@]}"; do
      jax_key="${entry%%:*}"
      oc_name="${entry##*:}"
      if [[ "$target" == "$jax_key" || "${target,,}" == "${oc_name,,}" ]]; then
        record_one "$jax_key" "$oc_name" 1 1
        found=1
        break
      fi
    done
    if [[ -z "$found" ]]; then
      echo "Unknown game: $target"
      echo "Choose: freeway montezumarevenge pong bankheist seaquest skiing | all | print"
      exit 1
    fi
    ;;
esac
