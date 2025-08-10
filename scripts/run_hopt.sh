#!/bin/bash

# Number of parallel jobs
N_JOBS=10
# Total evaluations = N_JOBS * max_evals_per_run (defined in the python script)
# 10 * 2 = 20 total evaluations

# Create results directory if it doesn't exist
mkdir -p results/hopt
mkdir -p logs/hopt_parallel

echo "Starting $N_JOBS sequential hyperparameter optimization jobs..."

for i in $(seq 1 $N_JOBS)
do
    TRIALS_FILE="results/hopt/trials_run_$i.pkl"
    LOG_FILE="logs/hopt_parallel/run_$i.log"
    
    echo "Launching job $i. Trials file: $TRIALS_FILE, Log file: $LOG_FILE"
    
    # Launch in foreground
    bash -c "source $HOME/conda/bin/activate tf2 && python main_hopt_unified.py --optimize --trials_file $TRIALS_FILE" > $LOG_FILE 2>&1
done

echo "All sequential jobs have completed."
echo "You can now run the collect_results.py script to find the best result."
