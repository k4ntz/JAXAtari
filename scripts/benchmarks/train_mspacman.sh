#!/bin/bash
# Train PQN or PPO agent for MsPacman (object-centric).
# Sets up a venv from scratch if needed — no conda required.
#
# Usage:
#   bash train_mspacman.sh pqn
#   bash train_mspacman.sh ppo

set -euo pipefail

# ─── Argument parsing ────────────────────────────────────────────────
if [ $# -ne 1 ] || { [ "$1" != "pqn" ] && [ "$1" != "ppo" ]; }; then
    echo "Usage: bash train_mspacman.sh <pqn|ppo>"
    exit 1
fi

ALG="$1"

if [ "$ALG" = "pqn" ]; then
    ALG_SCRIPT="pqn_agent.py"
    ALG_CONFIG="pqn_jaxatari_mspacman_object"
else
    ALG_SCRIPT="ppo_agent.py"
    ALG_CONFIG="ppo_jaxatari_mspacman_object"
fi

# ─── Resolve paths ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$SCRIPT_DIR"
mkdir -p logs

# ─── Environment setup ──────────────────────────────────────────────
# Load modules — adjust these to match your cluster
module purge
module load python/3.10
module load cuda/11.8

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install project + training deps if not already installed
if ! python -c "import jaxatari" 2>/dev/null; then
    echo "Installing jaxatari and dependencies..."
    pip install --upgrade pip
    pip install -e "$PROJECT_ROOT"
    pip install hydra-core omegaconf optax distrax "wandb[media]>=0.24.0" safetensors
    # Install JAX with CUDA support — adjust cuda version as needed
    pip install "jax[cuda11]"
    install-sprites
fi

echo "Python: $(which python)"
echo "JAX devices: $(python -c 'import jax; print(jax.devices())')"

# ─── Run training ───────────────────────────────────────────────────
LOG_FILE="logs/${ALG}_mspacman_object.log"
echo "========================================="
echo "Training: $ALG (object-centric) for MsPacman"
echo "Script:   $ALG_SCRIPT +alg=$ALG_CONFIG"
echo "Log:      $LOG_FILE"
echo "========================================="

python "$ALG_SCRIPT" "+alg=$ALG_CONFIG" 2>&1 | tee "$LOG_FILE"
