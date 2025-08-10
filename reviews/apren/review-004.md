# Review 004: Learning Rate Experiment Analysis

This review presents the analysis of the learning rate experiment conducted using `main_apren.py`. The experiment aimed to investigate the impact of varying learning rates on the convergence and accuracy of the PINN model in discovering the kinematic viscosity (`nu`) of the 2D Burgers' equation.

## Experiment Overview

The `main_apren.py` script executed 10 independent runs. In each run, a random learning rate (logarithmically scaled between $10^{-5}$ and $10^{-2}$) was used for the Adam optimizer. The model underwent a two-stage training process: an initial data-only pre-training phase (10,000 epochs) followed by a full loss training phase (2,000 Adam epochs and L-BFGS-B optimization). The true kinematic viscosity was set to $0.05$.

## Experiment Parameters

The following parameters were used consistently across all runs:

*   **Grid Points (x, y):** 41x41
*   **Time Steps:** 50
*   **True Kinematic Viscosity (`nu`):** 0.05
*   **Spatial Domain (x, y):** [0.0, 2.0] x [0.0, 2.0]
*   **Time Domain (t):** [0.0, 0.05]
*   **Neural Network Layers:** [3, 60, 60, 60, 60, 2] (3 input, 4 hidden layers with 60 neurons each, 2 output)
*   **Adam Epochs (Full Loss):** 2000
*   **Data-Only Pre-training Epochs:** 10000
*   **Data Loss Weight (`lambda_data`):** 1.0
*   **PDE Loss Weight (`lambda_pde`):** 1.0
*   **Number of PDE Collocation Points:** 60000
*   **L-BFGS-B Options:** `maxiter`: 100000, `maxfun`: 100000, `maxcor`: 100, `maxls`: 50, `ftol`: 1e-20

## Results

The table below summarizes the results for each of the 10 runs, including the randomly generated learning rate, the discovered `nu` value, and the relative error. The relative error is calculated as $|	ext{discovered } 
u - 	ext{true } 
u| / 	ext{true } 
u$.

| Run | Learning Rate      | Discovered $\nu$ | Relative Error |
|:----|:-------------------|:------------------|:---------------|
| 1   | 5.609627e-04       | 0.049084          | 0.0183         |
| 2   | 4.920569e-04       | 0.012321          | 0.7536         |
| 3   | 2.226416e-04       | 0.045081          | 0.0984         |
| 4   | 1.255771e-05       | 0.052413          | 0.0483         |
| 5   | 2.157846e-03       | 0.092456          | 0.8491         |
| 6   | 2.096136e-05       | 0.056127          | 0.1225         |
| 7   | 5.903032e-03       | 0.007521          | 0.8496         |
| 8   | 2.397242e-05       | 0.090610          | 0.8122         |
| 9   | 9.308454e-03       | 0.022440          | 0.5512         |
| 10  | 4.853310e-05       | 0.044057          | 0.1189         |

## Discussion of Results

The experiment reveals a significant sensitivity of the discovered kinematic viscosity (`nu`) to the initial learning rate of the Adam optimizer. While some learning rates (e.g., Run 1: 5.61e-04, Run 4: 1.26e-05) resulted in relatively low relative errors (1.83% and 4.83% respectively), others led to substantially higher errors, indicating poor convergence to the true `nu` value. This suggests that the choice of learning rate is critical for the accurate discovery of physical parameters in PINNs, especially when using a hybrid optimization approach.

There doesn't appear to be a simple linear relationship between the magnitude of the learning rate and the final accuracy. For instance, both a relatively high learning rate (Run 1) and a very low learning rate (Run 4) yielded good results, while intermediate and very high learning rates often performed poorly. This non-linear behavior is typical in neural network training and highlights the need for careful hyperparameter tuning.

## Profiling and HPC Analysis

*   **Data-Only Pre-training:** This phase consistently took approximately 15-20 seconds across all runs. This indicates a stable and efficient initial training stage, primarily focused on fitting the observed data.
*   **Adam Training (Full Loss):** The Adam optimization phase consistently ran for about 306-308 seconds for 2000 epochs. This phase is crucial for reducing the combined data and PDE loss and bringing the model closer to the optimal solution before the L-BFGS-B fine-tuning.
*   **L-BFGS-B Optimization:** The duration of the L-BFGS-B phase varied significantly, ranging from 79 seconds (Run 1) to 1391 seconds (Run 4). This variability is likely due to the different states of the neural network and the `log_nu_pinn` parameter at the end of the Adam phase, which in turn are influenced by the Adam learning rate. A more optimized state from Adam leads to faster convergence for L-BFGS-B, requiring fewer iterations (e.g., Run 1 had 112 iterations, while Run 4 had 788 iterations).

Overall, the L-BFGS-B phase, while powerful for fine-tuning, can be a bottleneck in terms of execution time if the Adam optimizer does not sufficiently prepare the network. The total training time for a single run ranged from approximately 400 seconds to over 1700 seconds, with the L-BFGS-B phase contributing significantly to the variance.

## Tuning Roadmap

Based on these results, the following tuning roadmap is proposed for improving the convergence and accuracy of `nu` discovery:

1.  **Learning Rate Schedule Optimization:** Instead of a fixed learning rate, implement a learning rate schedule (e.g., exponential decay, cosine annealing) for the Adam optimizer. This could help the model navigate the loss landscape more effectively, especially in the later stages of Adam training, leading to better initial conditions for L-BFGS-B.
2.  **Adaptive Learning Rate Algorithms:** Explore other adaptive learning rate algorithms beyond Adam, such as RMSprop or Adagrad, to see if they offer more stable and consistent convergence across different runs.
3.  **Hyperparameter Optimization Frameworks:** Utilize automated hyperparameter optimization frameworks (e.g., Optuna, Hyperopt) to systematically search for optimal learning rates and other hyperparameters (e.g., `lambda_data`, `lambda_pde`, number of neurons/layers). This would be more efficient than manual random sampling.
4.  **Ensemble Averaging:** Continue to use ensemble averaging (as explored in previous reviews) to mitigate the impact of individual run variability and obtain more robust `nu` predictions.
5.  **Further Profiling:** For future experiments, consider more detailed profiling (e.g., using TensorFlow's profiling tools) to identify specific computational bottlenecks within the Adam and L-BFGS-B phases, especially for runs with high L-BFGS-B durations.
6.  **Loss Function Refinement:** Investigate dynamic weighting of `lambda_data` and `lambda_pde` during training, or alternative loss formulations, to guide the optimization process more effectively towards accurate `nu` discovery.

These steps will contribute to a more robust and accurate PINN model for parameter discovery in the 2D Burgers' equation, with a focus on improving both precision and computational efficiency.

<br><sub>Last edited: 2025-08-21 06:07:07</sub>
