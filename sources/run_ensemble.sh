#!/bin/bash

# Activate conda environment
source $HOME/conda/bin/activate tf2

# Create results file and add header
RESULTS_FILE="results/scipy_ensemble_results_nu_0.05.csv"
echo "seed,discovered_nu,mse_u,mse_v,total_mse" > $RESULTS_FILE

# Loop through seeds and run experiment
for i in {1..3}
do
    echo "Running main_scipy.py with seed $i"
    # Run python script and capture output
    output=$(python src/main_scipy.py --seed $i)
    
    # Extract metrics from output
    discovered_nu=$(echo "$output" | grep "Final Discovered nu:" | awk '{print $4}')
    mse_u=$(echo "$output" | grep "Prediction MSE (u):" | awk '{print $4}')
    mse_v=$(echo "$output" | grep "Prediction MSE (v):" | awk '{print $4}')
    total_mse=$(echo "$output" | grep "Total Prediction MSE (u+v):" | awk '{print $5}')
    
    # Append results to CSV
    echo "$i,$discovered_nu,$mse_u,$mse_v,$total_mse" >> $RESULTS_FILE
    
    # Save full output to log file
    echo "$output" > logs/main_scipy_nu_0.05_seed_$i.txt
done
