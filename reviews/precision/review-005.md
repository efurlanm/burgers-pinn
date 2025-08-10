# Review 005: Analysis of Loss Weighting Experiments and New Plan

## 1. Summary of Loss Weighting Experiments

Four experiments were conducted to test the effect of the loss function weights (`lambda_data`, `lambda_pde`) on the precision of the discovered kinematic viscosity `nu`. The true value of `nu` is `0.05`.

| Run | `lambda_data` | `lambda_pde` | Discovered `nu` | Relative Error |
|:---:|:---:|:---:|:---|:---:|
| 1 | 20 | 1 | 0.039921 | 20.16% |
| 2 | 1 | 1 | 0.059917 | 19.83% |
| 3 | 1 | 20 | 0.060007 | 20.01% |
| 4 | 1 | 100 | 0.060041 | 20.08% |

**Conclusion**: The experiments show that modifying the loss weights alone does not significantly improve the accuracy of the parameter discovery. The relative error remains consistently around 20%. This suggests that other hyperparameters or aspects of the model are the limiting factors.

## 2. Re-evaluating the Approach

The initial hypothesis was that the loss weight balance was the primary issue. Since that is not the case, we must investigate other potential sources of imprecision. I will re-examine the provided literature and the script's configuration for new avenues of investigation.

Possible factors influencing the precision include:

1.  **Number of Adam Epochs**: We reduced this to 200 for speed. It's possible that the initial Adam optimization phase is not sufficient to get the weights into a good basin of attraction for the L-BFGS-B optimizer to find the true `nu`.
2.  **Neural Network Architecture**: The current architecture (`[3, 60, 60, 60, 60, 2]`) might not be optimal for this problem.
3.  **Number of PDE Collocation Points**: The number of points used to enforce the PDE residual (`80,000`) might be insufficient or excessive.
4.  **Learning Rate**: The Adam optimizer's learning rate is fixed at `0.001`.

## 3. Proposed Next Steps: Investigating Adam Epochs

As a next step, I will investigate the impact of the initial Adam training phase. A more thorough initial optimization might place the model in a better position for the more sensitive L-BFGS-B algorithm.

I will revert the loss weights to the most promising configuration so far (equal weights, `1.0` and `1.0`, from Run 2) and run a new set of experiments, varying only the number of Adam epochs.

**Experiment 5.1**: `adam_epochs = 1000` (original value)
**Experiment 5.2**: `adam_epochs = 5000`
**Experiment 5.3**: `adam_epochs = 10000`

This will help determine if a longer initial training phase can overcome the local minima problem and improve the final precision of `nu`.

<br><sub>Last edited: 2025-08-09 14:45:54</sub>
