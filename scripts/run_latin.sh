#!/bin/bash

# This script runs the latin hypercube experiment 5 times with different seeds
# to ensure the robustness of the results.

# Create results and logs directories
RESULTS_DIR="results/latin_ensemble"
LOGS_DIR="logs/latin"
mkdir -p $RESULTS_DIR
mkdir -p $LOGS_DIR

# Run experiment for 5 different seeds
for i in {1..5}
do
    echo "Running LATIN experiment with seed $i"
    (source $HOME/conda/bin/activate tf2 && python main_latin.py \
        --seed $i \
        --run_id "latin_seed_$i" \
        --results_dir $RESULTS_DIR) > "$LOGS_DIR/latin_run_seed_$i.txt" 2>&1
    
    # Check the exit code of the python script
    if [ $? -eq 0 ]; then
        echo "Finished LATIN experiment with seed $i successfully."
    else
        echo "Error in LATIN experiment with seed $i. Check log file: $LOGS_DIR/latin_run_seed_$i.txt"
    fi
done

echo "All LATIN experiments finished."
