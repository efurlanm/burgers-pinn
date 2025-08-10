import pickle
import glob
from hyperopt import hp
import numpy as np

def find_best_trial_across_runs():
    best_loss = float('inf')
    best_trial = None
    best_file = None

    # Define the exact same search space as in the main script
    space = {
        'seed': hp.choice('seed', [1, 17, 31, 42, 53, 61, 73, 89, 97, 123]),
        'neurons': hp.choice('neurons', [40, 50, 60, 70, 80]),
        'layers': hp.choice('layers', [4, 5, 6, 7]),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'adam_epochs_stage1': hp.choice('adam_epochs_stage1', [5000, 8000, 10000]),
        'epochs_data_only_stage1': hp.choice('epochs_data_only_stage1', [500, 1000]),
        'num_pde_points_stage1': hp.choice('num_pde_points_stage1', [10000, 20000]),
        'epochs_inverse_adam_pretrain': hp.choice('epochs_inverse_adam_pretrain', [1000, 1500, 2000]),
        'num_datasets_gene': hp.choice('num_datasets_gene', [10, 12, 15]),
        'noise_level': hp.uniform('noise_level', 0.0, 0.05),
        'nu_true': hp.uniform('nu_true', 0.01, 0.09)
    }
    # A small helper to get the actual values from the space definition
    # This is needed for hp.choice, which stores indices, not values.
    def get_choice_value(param_name, index):
        return space[param_name].pos_args[index+1].obj

    trial_files = glob.glob('results/hopt/trials_run_*.pkl')
    
    if not trial_files:
        print("No trial files found in results/hopt/. Make sure the parallel runs have completed.")
        return

    print(f"Found {len(trial_files)} trial files to analyze.")

    for file_path in trial_files:
        try:
            with open(file_path, 'rb') as f:
                trials = pickle.load(f)
        except Exception as e:
            print(f"Could not load or read {file_path}: {e}")
            continue
            
        for trial in trials:
            if 'result' in trial and 'loss' in trial['result'] and trial['result']['loss'] is not None:
                if trial['result']['loss'] < best_loss:
                    best_loss = trial['result']['loss']
                    best_trial = trial
                    best_file = file_path

    if best_trial:
        print("\n" + "="*50)
        print("Best overall trial found across all runs:")
        print(f"  Found in file: {best_file}")
        print(f"  Loss (Percentage Error): {best_trial['result']['loss']}%")
        print("  Parameters:")
        
        params = {key: value[0] for key, value in best_trial['misc']['vals'].items()}
        
        for key, value in params.items():
            # Check if the parameter is a choice type and map index to value
            if space[key].name == 'switch': # This is how hp.choice is identified
                 actual_value = get_choice_value(key, int(value))
                 print(f"    {key}: {actual_value}")
            else:
                 print(f"    {key}: {value}")
        print("="*50)
    else:
        print("No successful trials found across all runs.")

if __name__ == '__main__':
    find_best_trial_across_runs()
