#!/bin/bash

# Script to run the "PLATEAU" experiment (Base Case from CIACA 2025 paper)
# This experiment focuses on discovering kinematic viscosity (nu) for a single, fixed nu_true value.

echo "Activating conda environment 'tf2'..."
source $HOME/conda/bin/activate tf2

if [ $? -ne 0 ]; then
    echo "Error: Could not activate conda environment 'tf2'."
    exit 1
fi

# --- Default Parameters for the PLATEAU experiment ---
# These parameters are taken from the argparse defaults in scripts/main_plateau.py
# and also reflect the setup described in the CIACA 2025 paper.

NU_INITIAL=${1:-0.01} # Initial kinematic viscosity for curriculum training range
SEED=${2:-1}          # Random seed for reproducibility
NU_TRUE=${3:-0.05}    # True kinematic viscosity for data generation and inverse problem
NOISE_LEVEL=${4:-0.0} # Percentage of Gaussian noise (e.g., 0.01 for 1%)
ADAM_EPOCHS_STAGE1=${5:-5000} # Number of Adam epochs for Stage 1 (parametric PINN training)
EPOCHS_INVERSE_ADAM_STAGE2=${6:-5000} # Number of Adam epochs for Stage 2 (inverse problem, nu discovery)
EPOCHS_INVERSE_ADAM_PRETRAIN=${7:-1000} # Number of Adam pre-training epochs for Stage 2 inverse problem

# --- Output Configuration ---
# Results will be saved in burgers-pinn/results/plateau/
# Logs will be saved in burgers-pinn/logs/plateau/

RESULTS_DIR="results/plateau"
LOG_DIR="logs/plateau"
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${RESULTS_DIR}/plateau_nu_${NU_TRUE}_seed_${SEED}_${TIMESTAMP}.npz"
LOG_FILE="${LOG_DIR}/plateau_nu_${NU_TRUE}_seed_${SEED}_${TIMESTAMP}.log"

echo "--- Starting PLATEAU Experiment ---"
echo "Parameters:"
echo "  NU_INITIAL: $NU_INITIAL"
echo "  SEED: $SEED"
echo "  NU_TRUE: $NU_TRUE"
echo "  NOISE_LEVEL: $NOISE_LEVEL"
echo "  ADAM_EPOCHS_STAGE1: $ADAM_EPOCHS_STAGE1"
echo "  EPOCHS_INVERSE_ADAM_STAGE2: $EPOCHS_INVERSE_ADAM_STAGE2"
echo "  EPOCHS_INVERSE_ADAM_PRETRAIN: $EPOCHS_INVERSE_ADAM_PRETRAIN"
echo "Output will be saved to: $OUTPUT_FILE"
echo "Log will be saved to: $LOG_FILE"
echo "-----------------------------------"

python burgers-pinn/scripts/main_plateau.py \
    --nu_initial "$NU_INITIAL" \
    --seed "$SEED" \
    --nu_true "$NU_TRUE" \
    --noise_level "$NOISE_LEVEL" \
    --adam_epochs_stage1 "$ADAM_EPOCHS_STAGE1" \
    --epochs_inverse_adam_stage2 "$EPOCHS_INVERSE_ADAM_STAGE2" \
    --epochs_inverse_adam_pretrain "$EPOCHS_INVERSE_ADAM_PRETRAIN" \
    > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "PLATEAU experiment completed successfully. Check results in $RESULTS_DIR and logs in $LOG_DIR."
else
    echo "ERROR: PLATEAU experiment failed. Check log file: $LOG_FILE"
fi

echo "Deactivating conda environment..."
conda deactivate
