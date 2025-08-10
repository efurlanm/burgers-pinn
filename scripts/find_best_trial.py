import pickle

def find_best_trial_log():
    try:
        with open('results/hopt/hyperopt_trials.pkl', 'rb') as f:
            trials = pickle.load(f)
    except FileNotFoundError:
        print("Error: hyperopt_trials.pkl not found.")
        return

    best_trial = trials.best_trial
    best_trial_idx = trials.trials.index(best_trial)
    
    print(f"Best trial index: {best_trial_idx}")
    print(f"Corresponding log file: logs/hopt/trial_{best_trial_idx}.log")
    print(f"Loss (Percentage Error): {best_trial['result']['loss']}%")
    print("Parameters:")
    for key, value in best_trial['result']['params'].items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    find_best_trial_log()
