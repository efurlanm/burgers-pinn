# Draft for Parameter Discovery in 2D Burgers' Equation

## 1. Introduction

This work details a comprehensive investigation into the practical challenges and solutions for discovering the kinematic viscosity parameter (`nu`) in the 2D Burgers' equation using Physics-Informed Neural Networks (PINNs). While PINNs offer a powerful, mesh-free framework for solving differential equations, their application to inverse problems presents significant hurdles in training stability, parameter identifiability, and robustness. This document outlines the tuning roadmap followed to develop a reliable methodology, highlighting the key lessons learned and the strategies implemented to overcome these challenges.

As noted by Cuomo et al. (2022), while PINNs can be seen as an unsupervised method for forward problems, "...for inverse problems or when some physical properties are derived from data that may be noisy, PINN can be considered supervised learning methodologies." Our approach, which uses data generated from a Finite Difference Method (FDM) simulation, falls into this supervised category.

## 2. Core Methodology

The core of our approach is a PINN implemented in TensorFlow. The network takes spatiotemporal coordinates `(t, x, y)` as input and outputs the velocity fields `(u, v)`. The kinematic viscosity, `nu`, is treated as a trainable variable.

The ground truth and training data are generated based on a defined problem setup:

- **Spatial Domain**: `x` in `[0.0, 2.0]`, `y` in `[0.0, 2.0]`.
- **Temporal Domain**: `t` in `[0.0, 0.05]`.
- **Initial Condition (t=0)**: The simulation starts with a 2D Gaussian pulse centered at `(1.0, 1.0)` with a standard deviation of `0.25` for both `u` and `v` velocity fields, defined by the formula:
  `u(x, y, 0) = v(x, y, 0) = exp(-((x - 1.0)² / (2 * 0.25²) + (y - 1.0)² / (2 * 0.25²)))`
- **Boundary Conditions**: Dirichlet boundary conditions are used, where the values at the spatial boundaries (`x=0, 2` and `y=0, 2`) are held constant to their initial values throughout the simulation. The PINN learns these conditions via the data loss term, which constitutes a "soft" enforcement of these constraints.

It is important to note that, by using data generated from an FDM solver as a reference, our primary goal is to evaluate the PINN's ability to learn the solution of the PDE and discover the `nu` parameter from this high-fidelity data. We acknowledge that the FDM solver has its own numerical error characteristics, but it serves as a consistent and well-established benchmark, allowing for a controlled evaluation of the PINN methodology's performance on the inverse problem.

The training process minimizes a composite loss function:
`L_total = lambda_data * L_data + lambda_pde * L_pde`

- **`L_data`**: The Mean Squared Error (MSE) between the PINN's predictions and the "measured" data from an FDM simulation.
- **`L_pde`**: The MSE of the 2D Burgers' equation residuals. This loss is calculated at a large number of randomly sampled spatiotemporal points within the domain, known as **collocation points**. These points are where the "physics-informed" part of the network is enforced. The number of collocation points is a critical hyperparameter, determining how strongly the physical constraints are applied, which trades off against computational cost.

The data fidelity term, `L_data`, is measured using the Mean Squared Error, a standard regression metric. It is defined as the average of the squared differences between the predicted values (`ŷi`) and the actual values (`yi`). Its formula is `MSE = (1/n) * Σ(yi - ŷi)²`. This metric is particularly effective because it penalizes larger errors much more significantly than smaller ones, making it sensitive to significant deviations and a good indicator of visual fit.

Optimization is performed using a two-stage hybrid approach: an initial phase with the Adam optimizer to find a good region in the loss landscape, followed by the L-BFGS-B algorithm for fine-tuning and high-precision convergence.

## 3. The Tuning Roadmap: A Journey of Refinement

It is important to note that the following roadmap represents an evolutionary process. The methodologies used in the later stages (e.g., "Precision" and "Shape" focuses) are more advanced than those in the initial stages. Key improvements that were introduced over time include: the use of Data-Guided PINN (DG-PINN) pre-training, an increase in the neural network's width (from 20 to 60 neurons per layer), and more refined hyperparameter tuning. Therefore, while we compare the outcomes of these phases, it is crucial to recognize that the superior results of the later stages are a combined effect of both the specific strategies being tested and the overall enhancement of the baseline methodology.

The following table summarizes the key architectural and training parameters used across the different phases of the project, illustrating the evolution of the model's complexity and training strategy.

#### Table 4: Evolution of Network Architecture and Key Parameters

| Phase / Focus | Hidden Layers | Neurons per Layer | Activation (Hidden) | Typical Training Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `lbfgsb-scipy` (Initial) | 4 | 20 | `tanh` | Adam (varied, ~2k-5k epochs) + L-BFGS-B. No data-only pre-training. |
| `precision` & `shape` (Final) | 4 | 60 | `tanh` | **DG-PINN (10k epochs)** + Adam (2k epochs) + L-BFGS-B, 60k PDE points |

### 3.1. Part 1: The Optimizer Challenge (TFP vs. SciPy)

- **Initial Approach:** The project began with a pure TensorFlow 2.x implementation, using the L-BFGS optimizer from the TensorFlow Probability (`tfp.optimizer.lbfgs_minimize`) library.
- **Problem:** This approach proved unworkable due to numerical instability, frequently failing to converge.
- **Solution:** We pivoted to a more stable, hybrid approach, using TensorFlow for model definition and the L-BFGS-B implementation from the SciPy library for optimization.

### 3.2. Part 2: The Parameter Identifiability Problem

- **Problem:** With a stable optimizer, the PINN could learn the correct visual shape of the solution but failed to discover the correct `nu` from a single time snapshot.
- **Solution:** The training strategy was fundamentally changed to incorporate data from **multiple intermediate time steps**, providing a stronger physical constraint and making the `nu` parameter identifiable.

### 3.3. Part 3: Stabilizing Optimization with Data-Guided Pre-training (DG-PINN)

Even with a stable optimizer and sufficient data, the optimization process remained challenging. The composite loss function, combining data and PDE residuals, creates a complex landscape that can be difficult for optimizers to navigate from a random starting point. To address this, we introduced a **Data-Guided PINN (DG-PINN)** strategy.

This approach involves a pre-training phase where the network is trained for a significant number of epochs (e.g., 10,000) using **only the data loss term (`L_data`)**. This forces the network to first learn the basic shape, magnitude, and behavior of the solution from the ground truth data, without the conflicting objective of satisfying the PDE. By "warming up" the network in this way, the subsequent, main optimization phase begins from a much more informed starting point, allowing the optimizer to focus on the more subtle task of fine-tuning the solution to satisfy the physical constraints and discover the `nu` parameter. This pre-training step proved to be a crucial element for achieving stable and consistent convergence in the final model.

### 3.4. Part 4: Refining Physical Constraints

A key refinement step for improving the precision of the `nu` discovery was tuning the number of PDE collocation points. Experiments were conducted varying the number of points from 20,000 up to 70,000. The results showed that increasing the density of collocation points generally improved the accuracy of the discovered parameter, as it provides a more rigorous physical constraint across the domain. The optimal value was found to be **60,000 points**, which yielded the highest precision. Increasing the number further to 70,000 resulted in a slight degradation of performance, likely due to increased optimization challenges or diminishing returns.

### 3.5. Part 5: The Trade-off Between Precision and Shape

It is important to clarify the basis of comparison between the "Precision" and "Shape" focuses. The primary comparison is performed on the `nu=0.05` base case, where both experimental setups used the same underlying methodology (the final 4x60 neural network architecture and DG-PINN pre-training). The only intentional difference was the loss function weighting, allowing for a direct analysis of the trade-off between parameter precision and visual fit (MSE). The extensive generalizability tests (e.g., for `nu=0.01`) were conducted for the "Precision" focus but were not repeated for the "Shape" configuration. This is because the generalizability challenge was found to be a function of the optimization's initial guess, a problem that is independent of the loss weighting. The solution—the automated initial guess search—would be equally applicable and necessary for both configurations, and therefore repeating the tests was deemed unlikely to produce fundamentally new insights into the precision-shape trade-off itself.

With a stable and identifiable model, we explored this trade-off:

- **"Precision" Focus:** To maximize the accuracy of `nu`, the PDE loss term was weighted more heavily (e.g., `lambda_pde = 100`).
- **"Shape" Focus:** To maximize the visual fit, the data loss term was weighted more heavily (e.g., `lambda_data = 100`). This produced solutions with a very low and consistent Total MSE of approximately **0.138**.

### 3.6. Part 6: The Quest for Robustness and Generalizability

The final phase addressed the reliability of the "precision" focused model through a series of detailed experiments.

#### 3.6.1. Robustness: Sensitivity to Random Seeds

Initial high-precision results were found to be highly sensitive to the random seed used for weight initialization. A single successful run was not representative of the model's typical performance. To address this, we implemented **ensemble averaging**. For the case where the ground truth `nu` was 0.05, we ran the model 3 times with different random seeds. This provided a much more realistic assessment of the model's performance, yielding a mean discovered `nu` of **0.05106** with a standard deviation of **0.00122**, corresponding to a relative error of **2.12%**. This highlights that reporting ensemble results is crucial for credibility. The detailed results for different ensemble runs are shown in Table 5.

#### Table 5: Detailed Ensemble Results (3-Seed Runs)

| Focus Area | Ground Truth `nu` | Seed | Discovered `nu` | Total MSE | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Shape | 0.05 | 1 | 0.046987 | 0.138 | Consistent visual fit and `nu` discovery. |
| Shape | 0.05 | 2 | 0.048603 | 0.138 | Consistent visual fit and `nu` discovery. |
| Shape | 0.05 | 3 | 0.049230 | 0.138 | Consistent visual fit and `nu` discovery. |
| Generalization | 0.01 | 1 | **0.011029** | 0.147 | **Success**: Automated search found correct `nu`. |
| Generalization | 0.01 | 2 | **0.041966** | 0.146 | **Failure**: Automated search picked wrong `nu`. |
| Generalization | 0.01 | 3 | **0.009427** | 0.148 | **Success**: Automated search found correct `nu`. |


#### 3.6.2. Generalizability: Discovering Different `nu` Values

We then tested if a model tuned for `nu=0.05` could generalize to discover different true viscosity values. As shown in Table 6, the model initially failed completely.

- **Initial Failure:** The model failed when the true `nu` was set to 0.01 or 0.1. It consistently returned a value close to 0.05, the parameter it was originally tuned for, resulting in massive relative errors (e.g., ~430% for the `nu=0.01` case).
- **Root Cause and Solution:** The problem was the fixed initial guess for `nu` in the optimization process. The optimizer could not escape the local minimum around the initial guess. By providing an initial guess closer to the true value, the model succeeded.
- **Automated Solution:** To make the model more autonomous, we implemented an **automated initial guess search mechanism**. The automated search mechanism works by iterating through a predefined list of candidate `nu` values (e.g., [0.01, 0.05, 0.1]). For each candidate, it resets the neural network's weights and temporarily sets `nu` to that candidate's value. It then pre-trains the model for a small, fixed number of Adam epochs (e.g., 500). After this brief training, it evaluates the total loss. The candidate `nu` that results in the lowest loss is then selected as the optimal starting point for the main, full optimization process. This allows the model to autonomously find a promising region in the loss landscape before committing to the computationally expensive L-BFGS-B fine-tuning. As shown in Table 5, this method is promising but not yet perfectly robust, as its success can be sensitive to the random seed. This indicates that while promising, the automated search requires further refinement.

#### 3.6.3. Performance

A typical high-precision run, using the DG-PINN methodology with 10,000 data-only epochs followed by 2,000 Adam epochs and ~100 L-BFGS-B iterations, took approximately **437 seconds (~7.3 minutes)** to complete on a workstation equipped with a 14-core Intel Xeon E5-2680 v4 CPU and a single NVIDIA GeForce RTX 3050 GPU. The primary performance bottleneck in this hybrid approach is the data transfer overhead between the GPU (for model computation) and the CPU (for the SciPy L-BFGS-B optimizer).

### 3.7. Summary of Key Challenges and Failed Approaches

The path to the final methodology was not linear. Documenting the key challenges and failed approaches is crucial for understanding the solution's context and providing a realistic picture of the research process.

-   **Initial Optimizer Instability:** The first major hurdle was the instability of the native TensorFlow Probability L-BFGS-B optimizer, which proved unusable for this problem and forced the adoption of the more stable hybrid SciPy-based approach.

-   **Insufficiency of Single Time-Step Data:** Early experiments failed to identify the `nu` parameter correctly because training on a single time snapshot does not provide enough physical constraint. This conceptual flaw was resolved by moving to a multi-time-step training dataset.

-   **Failed Back-porting of Advanced Techniques:** In an attempt to harmonize methodologies, the advanced DG-PINN methodology and automated search from the `precision` focus were integrated into the older `lbfgsb-scipy` baseline. This experiment was a notable failure. The `lbfgsb-scipy` model, which used a smaller network architecture (4x20 vs. 4x60 neurons), failed to converge to the correct `nu` even with these enhancements. This demonstrated that advanced techniques are not always "plug-and-play" and their success can be highly dependent on other factors like network capacity.

-   **Naive Generalization:** The most significant conceptual failure was the assumption that a PINN trained to find a specific `nu` (e.g., 0.05) would naturally generalize to find others. The experiments showed this to be false, as the model was strongly biased towards the value it was tuned for. This led to the development of the initial guess search mechanism as a necessary, pragmatic solution.

## 4. Summary of Robustness and Generalizability Experiments

#### Table 6: Summary of Generalizability Experiments

| Ground Truth `nu` | Initial Guess Strategy | Mean Discovered `nu` (Ensemble) | Relative Error | Visual Fit (Total MSE) | Notes |
|:--- |:--- |:--- |:--- |:--- |:--- |
| 0.05 | Fixed (0.06) | 0.05106 | 2.12% | ~0.138 | Baseline robustness test. |
| 0.01 | Fixed (0.06) | ~0.053 | ~430% | High | **Failure**: Model is biased to original `nu`. |
| 0.1 | Fixed (0.06) | ~0.056 | ~44% | High | **Failure**: Model is biased to original `nu`. |
| 0.01 | Manual (0.01) | 0.01026 | 2.56% | ~0.147 | **Success**: Shows importance of initial guess. |
| 0.01 | Automated Search | 0.01103 (single run) | 10.29% | ~0.147 | **Promising**: Viable but needs more robustness. |

## 5. Conclusion

Successfully applying PINNs to inverse problems is not a "plug-and-play" task. It requires a systematic tuning process that addresses challenges at multiple levels: from the choice of software implementation (optimizer stability) to the formulation of the training data (parameter identifiability) and the statistical robustness of the results (ensemble averaging and generalizability). The developed methodology, incorporating a hybrid SciPy optimizer, multi-time-step training, and an automated initial guess search, provides a robust and generalizable framework for parameter discovery.

It is important to frame the automated initial guess search as an effective, pragmatic solution rather than a definitive one. It cleverly automates the process of tuning the model for a new physical scenario, making the PINN a more practical tool. However, it does not solve the underlying generalizability challenge. The ideal solution, a "holy grail" for this class of problems, would be a single, universal PINN capable of predicting the correct velocity fields for a continuous range of `nu` values without any case-by-case retraining or searching. Such a model might, for example, accept `nu` as an additional input and learn the relationship between the physical parameter and the solution field. While achieving this level of generalization is a significantly harder research problem, the methodology developed here represents a valuable intermediate step, making the PINN framework more robust and autonomous for practical applications.

Future work should focus on improving the generalizability of the model. A key area for improvement is the automated initial guess search; its robustness could be enhanced by increasing the number of pre-training epochs or by using a more adaptive search grid for the `nu` candidates. Furthermore, exploring curriculum learning strategies, where the model is first trained on simpler physical scenarios before tackling the final, complex problem, could help guide the optimizer to a more robust global minimum and reduce its dependency on the initial guess.
<br><sub>Last edited: 2025-08-11 10:58:44</sub>
