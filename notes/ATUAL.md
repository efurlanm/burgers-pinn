# Current Status Report: Parameter Discovery in 2D Burgers' Equation using PINNs

## 1. Introduction

This report details the current progress in applying Physics-Informed Neural Networks (PINNs) for the discovery of the kinematic viscosity parameter ($\nu$) in the 2D Burgers' equation. The 2D Burgers' equation is a fundamental partial differential equation (PDE) in fluid dynamics, serving as a simplified model for turbulence and shock waves. Its system of two coupled PDEs is given by:

1. For the $u$ component (velocity in x-direction):
   $u_t + u \cdot u_x + v \cdot u_y - \nu \cdot (u_{xx} + u_{yy}) = 0$
2. For the $v$ component (velocity in y-direction):
   $v_t + u \cdot v_x + v \cdot v_y - \nu \cdot (v_{xx} + v_{yy}) = 0$

Where $u$ and $v$ are velocity components, and subscripts denote partial derivatives with respect to time ($t$) and spatial coordinates ($x, y$). The parameter $\nu$ represents the kinematic viscosity, which is the target for discovery in these experiments.

The primary objective of this work is to evaluate different PINN architectures and training strategies, with a particular focus on generalizing the discovered parameter and optimizing High-Performance Computing (HPC) aspects. This report specifically analyzes three main experimental setups: `main_precision.py`, `main_shape.py`, and `main_prmtrc.py`.

## 2. Methodology

All three PINN implementations share a common foundation: they leverage TensorFlow 1.x for neural network construction and automatic differentiation, and employ a hybrid optimization strategy combining Adam (for initial convergence) and SciPy's L-BFGS-B (for fine-tuning). Ground truth data for training and validation is generated internally using a TensorFlow-based Finite Difference Method (FDM) solver for the 2D Burgers' equation.

### 2.1. `main_precision.py` (Precision-Focused PINN)

This script represents a baseline approach focused on precisely discovering a single, fixed kinematic viscosity parameter.

* **Neural Network Input**: The neural network takes spatial coordinates and time as input: $(x, y, t)$.
* **Discoverable Parameter**: The kinematic viscosity $\nu$ is treated as a single trainable variable, initialized as $\log(\nu_{pinn})$ for numerical stability.
* **Activation Function**: Tanh activation is used in the hidden layers.
* **Loss Function**: The total loss is a weighted sum of data fidelity loss (Mean Squared Error between predicted and true velocities) and PDE residual loss (Mean Squared Error of the Burgers' equation residuals). The weights ($\\lambda_{data}$, $\\lambda_{pde}$) are fixed hyperparameters (e.g., 1.0 for data, 1.0 for PDE in the analyzed log `precision_run_dg_pinn_with_mse.txt`).
* **Training Strategy**:
  1. **Data-Only Pre-training**: An initial phase where only the data loss is minimized using the Adam optimizer.
  2. **Full Adam Optimization**: Training with the combined data and PDE losses using Adam.
  3. **L-BFGS-B Optimization**: Fine-tuning the model parameters and $\nu$ using the L-BFGS-B optimizer.
* **Initial $\nu$ Guess**: Includes a mechanism to find the "best" initial $\nu$ from a set of candidates by performing short pre-training runs and selecting the one with the lowest loss.

### 2.2. `main_shape.py` (Shape-Focused PINN)

This script is a variation of `main_precision.py`, primarily distinguished by its emphasis on fitting the shape of the data more accurately.

* **Neural Network Input**: Similar to `main_precision.py`, the input is $(x, y, t)$.
* **Discoverable Parameter**: $\nu$ is a single trainable parameter.
* **Activation Function**: Tanh activation is used in the hidden layers.
* **Loss Function**: The key difference lies in the loss weighting. The data fidelity loss weight ($\\lambda_{data}$) is significantly higher (e.g., 100.0) compared to the PDE residual loss weight ($\\lambda_{pde}$, e.g., 1.0). This prioritizes fitting the observed data points.
* **Training Strategy**: Follows the same three-stage optimization as `main_precision.py` (Data-Only Adam, Full Adam, L-BFGS-B).
* **Initial $\nu$ Guess**: Does not include the `find_best_initial_nu` method; it starts with a fixed initial guess for $\nu$.

### 2.3. `main_prmtrc.py` (Parametric PINN for Generalization)

This script introduces a more advanced approach aimed at generalizing the PINN to discover $\nu$ across a range of values, rather than just a single fixed value.

* **Neural Network Input**: The neural network takes $(x, y, t, \nu)$ as input. This allows the network to learn a mapping from input coordinates and a given $\nu$ to the corresponding velocity fields.
* **Discoverable Parameter**:
  * **Stage 1 (Parametric Training)**: The network is trained to predict $u$ and $v$ for a *range* of $\nu$ values (e.g., [0.01, 0.1]). The $\nu$ input to the network is sampled from this range.
  * **Stage 2 (Inverse Problem)**: After Stage 1, the trained network's weights are frozen. A *new* trainable parameter, $\log(\nu_{inverse})$, is introduced. This stage focuses on discovering a *specific* $\nu$ value by minimizing the MSE between the network's predictions (using the frozen parametric network and the discoverable $\nu_{inverse}$) and a given set of observed data.
* **Activation Function**: Swish activation ($x \cdot \text{sigmoid}(x)$) is used in the hidden layers.
* **Loss Function**:
  * **Adaptive Loss Weighting**: $\\lambda_{data}$ and $\\lambda_{pde}$ are initialized as *trainable TensorFlow variables*. This allows the model to dynamically adjust the weights of the data and PDE losses during training, aiming to balance their contributions and improve convergence. This adaptive weighting strategy was explored as an insight from the `PINNSTRIPES` repository, which investigates regularization and adaptive weighting techniques.
  * **Learning Rate Schedule**: An exponential decay learning rate schedule is applied to the Adam optimizer in Stage 1.
* **Training Strategy**: A two-stage optimization process:
  1. **Stage 1 (Parametric PINN Training)**: Includes Data-Only Adam, Full Adam (with dynamic PDE point generation in batches), and L-BFGS-B. The goal is to train a robust network capable of handling varying $\nu$ inputs.
  2. **Stage 2 (Inverse Problem)**: A separate Adam optimization phase where only $\nu_{inverse}$ is trained, using the pre-trained parametric network.

## 3. Experimental Results and Discussion

This section presents the key results from representative runs of each script, focusing on the discovered $\nu$ values, prediction Mean Squared Errors (MSEs), and the behavior of the training process.

### 3.1. `main_prmtrc.py` Analysis (Parametric PINN)

The analyzed log for `main_prmtrc.py` (`parametric_inverse_run_seed_1_attempt_29_annealing_weights.txt`) provides a detailed view of its performance. The ground truth $\nu$ for the inverse problem was 0.05.

* **Stage 1 (Parametric Training)**:
  
  * Data-Only Pre-training (10000 epochs): Achieved a data loss of approximately $2 \times 10^{-6}$. This phase effectively initialized the network to fit the data.
  * Adam Training (5000 epochs): The total loss decreased significantly, and the adaptive weights ($\\lambda_{data}$, $\\lambda_{pde}$) showed dynamic adjustment. For instance, $\\lambda_{data}$ decreased from ~0.99 to ~0.77, while $\\lambda_{pde}$ decreased from ~0.99 to ~0.84. This suggests the annealing mechanism was active, attempting to balance the loss contributions.
  * L-BFGS-B Training: Further refined the model.
  * Prediction MSE (for $\nu=0.05$ using Stage 1 model):
    * MSE (u): $4.379974 \times 10^{-2}$
    * MSE (v): $4.296395 \times 10^{-2}$
    * Total MSE (u+v): $8.676368 \times 10^{-2}$
      These MSEs indicate the model's ability to predict velocities for a specific $\nu$ within the trained range after Stage 1.

* **Stage 2 (Inverse Problem - Discovering $\nu$)**:
  
  * Training (5000 epochs): The `nu_inverse` parameter was optimized.
  * **Discovered $\nu$**: $0.001664$
  * **Ground Truth $\nu$**: $0.05$
  * Prediction MSE (for Discovered $\nu=0.001664$):
    * MSE (u): $6.876207 \times 10^{-4}$
    * MSE (v): $6.871482 \times 10^{-4}$
    * Total MSE (u+v): $1.374769 \times 10^{-3}$

**Discussion**: While the Stage 1 model achieved reasonable prediction MSEs for a given $\nu$, the discovered $\nu$ in Stage 2 ($0.001664$) is significantly off from the ground truth ($0.05$). This indicates that despite the parametric training and adaptive weighting, the model struggled to accurately infer the specific kinematic viscosity in the inverse problem. The low MSEs in Stage 2 for the *discovered* $\nu$ suggest that the model found a $\nu$ value that minimizes the data loss for the given observations, but this value is not the true $\nu$. This could be due to several factors, including:
    *   **Identifiability Issues**: The Burgers' equation might have multiple $\nu$ values that produce similar velocity fields under certain conditions, making it hard for the PINN to uniquely identify the true $\nu$.
    *   **Loss Landscape**: The loss landscape for $\nu$ might be very flat or have local minima, trapping the optimizer.
    *   **Data Quantity/Quality**: The amount or distribution of observed data for the inverse problem might be insufficient to constrain $\nu$ effectively.
    *   **Network Capacity**: The network might not have sufficient capacity to learn the complex relationship between input $\nu$ and the velocity fields accurately enough for precise inverse problem solving.
    *   **Adaptive Weighting Effectiveness**: While adaptive weights were used, their specific implementation or annealing schedule might not be optimal for this problem.

### 3.2. `main_precision.py` Analysis (Precision-Focused PINN)

The `precision_run_dg_pinn_with_mse.txt` log shows a run with fixed loss weights ($\\lambda_{data}=1.0$, $\\lambda_{pde}=1.0$). The ground truth $\nu$ was 0.05.

* **Initial $\nu$ Guess**: $0.060000$
* **Final Discovered $\nu$**: $0.050671$
* **Prediction MSE (for $\nu=0.05$)**:
  * MSE (u): $6.846815 \times 10^{-2}$
  * MSE (v): $6.852602 \times 10^{-2}$
  * Total MSE (u+v): $1.369942 \times 10^{-1}$

**Discussion**: The discovered $\nu$ ($0.050671$) is very close to the true value ($0.05$), indicating good performance in parameter discovery for this setup. The prediction MSEs for $u$ and $v$ are also provided, allowing for a more complete assessment of the model's accuracy in predicting the velocity field.

### 3.3. `main_shape.py` Analysis (Shape-Focused PINN)

The `main_shape_nu_0.05_seed_1.txt` log highlights the impact of a very high data loss weight ($\\lambda_{data}=100.0$, $\\lambda_{pde}=1.0$). The ground truth $\nu$ was 0.05.

* **Initial $\nu$ Guess**: $0.060000$
* **Final Discovered $\nu$**: $0.046987$
* **Prediction MSE (for $\nu=0.05$)**:
  * MSE (u): $6.909595 \times 10^{-2}$
  * MSE (v): $6.909173 \times 10^{-2}$
  * Total MSE (u+v): $1.381877 \times 10^{-1}$

**Discussion**: The discovered $\nu$ ($0.046987$) is the closest to the true value ($0.05$) among the analyzed runs, suggesting that a strong emphasis on data fitting can indeed help in parameter discovery for this specific setup. However, the overall prediction MSEs for $u$ and $v$ are relatively high ($1.38 \times 10^{-1}$), indicating that while the model found a good $\nu$, its ability to accurately predict the full velocity field might be compromised by the strong data constraint, potentially leading to less adherence to the PDE. This highlights a common trade-off in PINNs between data fidelity and PDE satisfaction.

### 3.4. `main_plateau.py` Analysis (ReduceLROnPlateau)

The `plateau_run_seed_1.txt` log details the performance of the `main_plateau.py` script, which incorporates the `ReduceLROnPlateau` learning rate scheduler. The ground truth $\nu$ for the inverse problem was 0.05.

*   **Stage 1 (Parametric Training)**:
    *   **Data-Only Pre-training (10000 epochs)**: Achieved a data loss of approximately $2 \times 10^{-6}$. Duration: 26.59 seconds.
    *   **Adam Training (5000 epochs)**: The total loss decreased from $0.275752$ to $0.000023$. However, the `ReduceLROnPlateau` mechanism did not trigger, suggesting continuous (though potentially slow) loss improvement or a high patience setting. Duration: 1553.39 seconds.
    *   **L-BFGS-B Training**: Converged successfully (`L-BFGS-B converged: True`) in 46.61 seconds, indicating effective pre-training by Adam.
    *   **Prediction MSE (for $\nu=0.05$ using Stage 1 model)**:
        *   MSE (u): $4.373699 \times 10^{-2}$
        *   MSE (v): $4.284945 \times 10^{-2}$
        *   Total MSE (u+v): $8.658645 \times 10^{-2}$

*   **Stage 2 (Inverse Problem - Discovering $\nu$)**:
    *   **Training (5000 epochs)**: The `nu_inverse` parameter was optimized.
    *   **Discovered $\nu$**: $0.001476$
    *   **Ground Truth $\nu$**: $0.05$
    *   **Prediction MSE (for Discovered $\nu=0.001476$)**:
        *   MSE (u): $7.011628 \times 10^{-4}$
        *   MSE (v): $6.999635 \times 10^{-4}$
        *   Total MSE (u+v): $1.401126 \times 10^{-3}$

**Discussion**: While the Stage 1 Adam training successfully reduced the overall loss and enabled L-BFGS-B convergence, the `ReduceLROnPlateau` did not activate. More critically, the discovered $\nu$ in Stage 2 ($0.001476$) remains significantly different from the ground truth ($0.05$). This suggests that even with a more robust Stage 1 training, the model still struggles with accurate inverse parameter discovery. The low MSE for the *discovered* $\nu$ in Stage 2 indicates that the model found a $\nu$ that minimizes the data loss for the given observations, but this value does not correspond to the true physical parameter. This highlights the ongoing challenge of parameter identifiability and the need for further refinement in the parametric PINN's ability to infer physical parameters accurately.

## 4. Performance Analysis

This section compares the computational performance of the three PINN implementations based on the provided log files. All durations are in seconds.

| Metric / Model                           | `main_prmtrc.py` (Parametric) | `main_precision.py` (Baseline) | `main_shape.py` (High Data Weight) | `main_plateau.py` (ReduceLROnPlateau) |
|:---------------------------------------- |:----------------------------- |:------------------------------ |:---------------------------------- |:-------------------------------------- |
| Data Preparation Duration                | 7.23                          | 3.55                           | 3.51                               | 6.74                                   |
| Model Initialization Duration            | 9.22                          | 3.62                           | 3.53                               | 8.78                                   |
| Data-Only Pre-training Duration          | 28.26                         | 21.58                          | 20.86                              | 26.59                                  |
| Adam Training Duration (Stage 1)         | 1889.81                       | 107.11                         | 154.66                             | 1553.39                                |
| L-BFGS-B Training Duration (Stage 1)     | 33.29                         | 195.03                         | 16.57                              | 46.61                                  |
| Inverse Adam Training Duration (Stage 2) | 20.95                         | N/A                            | N/A                                | 20.65                                  |
| **Total Execution Duration**             | **1995.16**                   | **333.03**                     | **200.10**                         | **1669.04**                            |

**Discussion on Computational Cost:**

* **`main_prmtrc.py` (Parametric)**: This model has the highest computational cost, primarily due to its extensive Stage 1 Adam training (1889.81 seconds for 5000 epochs). This is expected, as the network is learning a more complex, generalized mapping across a range of $\nu$ values, and it also involves dynamic PDE point generation. The additional Stage 2 for inverse problem solving adds further overhead.
* **`main_plateau.py` (ReduceLROnPlateau)**: This model also has a high computational cost (1669.04 seconds), primarily driven by its Stage 1 Adam training (1553.39 seconds). While the `ReduceLROnPlateau` mechanism was implemented, it did not activate during this run, suggesting that the loss was continuously improving or the patience setting was too high. Its total duration is comparable to `main_prmtrc.py` due to the similar extensive Adam training.
* **`main_shape.py` (High Data Weight)**: This model is significantly faster than the parametric models, with a total execution duration of 200.10 seconds. Its Adam training phase is much shorter (154.66 seconds) compared to `main_prmtrc.py` and `main_plateau.py`. This suggests that focusing on a single $\nu$ and potentially the fixed PDE points (as opposed to dynamic generation) contributes to faster convergence in terms of training time.
* **`main_precision.py` (Baseline)**: This model has a total execution duration of 333.03 seconds. While its Adam training is shorter than `main_shape.py`, its L-BFGS-B phase is considerably longer. This highlights that the choice of optimizer and its parameters can significantly impact the overall training time.

**Tuning Roadmap:**

1. **Reduce Adam Epochs for Parametric Model**: The 5000 Adam epochs in Stage 1 of `main_prmtrc.py` and `main_plateau.py` contribute significantly to their long training times. Investigate if fewer epochs are sufficient, perhaps by monitoring validation loss or discovered $\nu$ convergence.
2. **Optimize PDE Point Generation**: The dynamic generation of PDE points in `main_prmtrc.py` and `main_plateau.py` might add overhead. Explore strategies like generating a larger fixed set of PDE points once, or using more efficient sampling methods if dynamic generation is critical.
3. **Batch Size Optimization**: Experiment with different batch sizes for Adam training. Larger batch sizes can sometimes lead to faster training per epoch on GPUs, but might require more epochs for convergence.
4. **Learning Rate Schedule Tuning**: Fine-tune the learning rate decay parameters for Adam in `main_prmtrc.py` and `main_plateau.py`. An aggressive decay might speed up convergence but risk getting stuck in local minima. For `main_plateau.py`, specifically investigate why `ReduceLROnPlateau` did not activate and adjust its parameters (`patience`, `factor`) accordingly.
5. **L-BFGS-B Options**: Review the `options` for `scipy.optimize.minimize` (e.g., `maxiter`, `maxfun`, `ftol`). Tighter tolerances can lead to longer training times.
6. **Hardware Acceleration**: Ensure that TensorFlow is fully utilizing available GPUs.
7. **Code Profiling**: Use TensorFlow's profiling tools to identify bottlenecks in the computation graph.
8. **Adaptive Weighting Strategy**: While adaptive weighting is promising, its implementation can impact performance. Investigate alternative adaptive weighting schemes (e.g., those from PINNSTRIPES) or simpler fixed weighting if the benefits of adaptivity do not outweigh the computational cost or lead to poor $\nu$ discovery.

## 5. Conclusion

The experiments demonstrate varying degrees of success in discovering the kinematic viscosity $\nu$ for the 2D Burgers' equation using PINNs.

* The **`main_shape.py`** model, with its strong emphasis on data fidelity (high $\\lambda_{data}$), achieved the closest $\nu$ discovery to the true value ($0.046987$ vs $0.05$). However, this came at the cost of higher overall prediction MSEs for the velocity fields, suggesting a trade-off where fitting the data very closely might lead to less adherence to the PDE.
* The **`main_precision.py`** model showed good $\nu$ discovery ($0.050671$ vs $0.05$) and provided comprehensive MSE metrics for velocity prediction.
* The **`main_prmtrc.py`** (parametric) model, despite its sophisticated two-stage training, adaptive loss weighting, and generalization capabilities, struggled significantly in accurately discovering the specific $\nu$ in its inverse problem stage ($0.001664$ vs $0.05$). This indicates that while the model might learn a generalized mapping, inferring a precise parameter from limited observed data using a frozen network remains a challenge. The high computational cost of this model is also a concern.
* The **`main_plateau.py`** model, incorporating `ReduceLROnPlateau`, also exhibited a significant discrepancy in discovered $\nu$ ($0.001476$ vs $0.05$) in its inverse problem stage, similar to `main_prmtrc.py`. While its Stage 1 Adam training was thorough, the `ReduceLROnPlateau` did not activate, and the model still struggled with accurate inverse parameter discovery. This reinforces the challenges of parameter identifiability and the need for further refinement in the parametric PINN's ability to infer physical parameters accurately.

**Key Takeaways and Future Work:**

* **Parameter Identifiability**: The difficulty in accurately discovering $\nu$ in the inverse problem, especially for the parametric models (`main_prmtrc.py` and `main_plateau.py`), suggests potential identifiability issues or a complex loss landscape. Further investigation into the sensitivity of the Burgers' equation solution to $\nu$ and the design of the inverse problem setup is warranted.
* **Balancing Losses**: The trade-off between data loss and PDE loss is crucial. While `main_shape.py` showed that strong data weighting can help $\nu$ discovery, it might hurt overall solution accuracy. Adaptive weighting in `main_prmtrc.py` is a promising direction, but its effectiveness needs to be rigorously evaluated and potentially refined.
* **Generalization vs. Precision**: The parametric models aim for generalization but currently struggle with precision in inverse problems. Future work should focus on improving the inverse problem stage of the parametric models, perhaps by:
  * Allowing some fine-tuning of the network weights in Stage 2, alongside $\nu_{inverse}$.
  * Using more diverse or strategically sampled data for the inverse problem.
  * Exploring different network architectures or regularization techniques.
* **Optimization**: The parametric models' high computational cost needs to be addressed. The tuning roadmap outlined in Section 4 provides concrete steps for optimizing performance. For `main_plateau.py`, specifically, understanding why `ReduceLROnPlateau` did not activate and adjusting its parameters will be crucial.
* **Quantitative Comparison**: For a more robust comparison, it is essential to systematically run all models with consistent configurations (e.g., same number of epochs, PDE points, seeds) and extract all relevant metrics (discovered $\nu$, MSEs for $u$ and $v$, and all durations) from their respective `.npz` files. This will allow for a more precise and fair evaluation.
* **PINNSTRIPES Insights**: The adaptive weighting strategy in `main_prmtrc.py` was inspired by insights from the `PINNSTRIPES/` repository. Further investigation into their regularization and adaptive weighting strategies could provide valuable insights for improving the current models.
<br><sub>Last edited: 2025-08-22 06:02:34</sub>
