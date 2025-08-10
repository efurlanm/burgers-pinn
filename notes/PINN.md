## Knowledge Base: Physics-Informed Neural Networks (PINNs) and `main_precision.py`

This document details the concepts and implementation within the `main_precision.py` script for Physics-Informed Neural Networks (PINNs), specifically focusing on a 2D Burgers' equation problem.

### I. Introduction to PINNs and `main_precision.py` Context

Physics-Informed Neural Networks (PINNs) are a class of neural networks that integrate governing physical laws (expressed as Partial Differential Equations, PDEs) into their training process, alongside observational data. This allows PINNs to learn solutions that are consistent with both data and physics.

in the context of numerical simulations and PINNs, the dimensions typically refer to the **spatial grid resolution**, indicating the number of discrete points used to represent a 2D spatial domain (e.g., 41 points along the x-axis and 41 points along the y-axis). They can also represent **data structure size** for storing computed values (like velocity components u and v) at each grid point, or **visualization resolution**. Higher numbers signify finer resolution, potentially leading to more accurate results but at a higher computational cost.

The `main_precision.py` script is designed to:

* Generate ground truth data using a Finite Difference Method (FDM) solver.
* Train a PINN to solve the 2D Burgers' equation, incorporating both data fidelity and PDE residual minimization.

### II. Data Generation for Ground Truth and PINN Training

The `main_precision.py` script meticulously defines different sets of points for various purposes:

* **Spatial Discretization for Ground Truth Data Generation (FDM Solver)**:
  
  * `grid_points_x = 41`
  * `grid_points_y = 41`
  * This means the FDM solver uses a **spatial grid of 41x41 points** to generate the "ground truth" solution.

* **Temporal Discretization for Ground Truth Data Generation (FDM Solver)**:
  
  * `time_steps = 50` (referred to as `nt`)
  * `dt = 0.001` (defined implicitly)
  * The FDM solver simulates for 50 time steps, resulting in a total simulation time of `time_steps * dt = 50 * 0.001 = 0.05`.

* **Ground Truth Data Snapshots**:
  
  * Ground truth data (u and v velocity fields) is **saved at specific time steps** of the FDM simulation to create a dataset for PINN training.
  * These snapshots are taken at `nt/4`, `nt/2`, `3*nt/4`, and `nt`.
  * Given `nt = 50` and the use of the `int()` function for truncation:
    * `int(50 / 4) = 12`
    * `int(50 / 2) = 25`
    * `int(3 * 50 / 4) = 37`
    * `50 = 50`
  * Therefore, snapshots are taken at **time steps 12, 25, 37, and 50**. These snapshots provide "ground truth" data points across the temporal evolution of the Burgers' equation.

* **PINN Data Points (for Data Loss)**:
  
  * These points are derived from the **flattened 41x41 FDM grid at the 4 saved time snapshots**.
  * The total number of data points for the PINN's data loss is `41 * 41 * 4 = 6724`.
  * These 6724 points are stored in `x_data_tf`, `y_data_tf`, `t_data_tf`, `u_data_tf`, and `v_data_tf` tensors.
  * They are used for the **data fidelity loss** where the PINN tries to match known u and v values.

* **PINN PDE Collocation Points (for PDE Loss)**:
  
  * `num_pde_points = 60000`
  * These are **60,000 random collocation points** generated within the spatio-temporal domain defined by `x_min`, `x_max`, `y_min`, `y_max`, `t_min`, `t_max`.
  * **They are generated independently** from the 6724 data points and are **not calculated from them**.
  * **Generation Code Explanation**:
    * `x_min, x_max = 0.0, 2.0`
    * `y_min, y_max = 0.0, 2.0`
    * `t_min = 0.0`, `t_max = time_steps * 0.001 = 0.05`.
    * `np.random.uniform(low, high, size=(num_pde_points, 1))` is used to generate 60,000 random numbers from a uniform distribution for each coordinate (x, y, t) within their respective bounds.
    * These NumPy arrays are then converted into TensorFlow constant tensors (`x_pde`, `y_pde`, `t_pde`).
  * **Fixed Throughout Training**: These 60,000 collocation points are generated **only once** at the beginning of the script and remain **fixed and unchanged** throughout the entire training process (Adam and L-BFGS-B optimization phases).
  * Their purpose is to enforce the physical laws (the PDE) during training through the **PDE residual loss**.

### III. PINN Training Process and Loss Function

The PINN in `main_precision.py` is trained by minimizing a total loss function that combines two primary components:

* **1. Data Fidelity Loss (`loss_data`)**:
  
  * **Purpose**: To ensure the neural network's predictions accurately match the available observed or "ground truth" data.
  * **Mechanism**:
    1. **Input Data Points**: The 6724 spatio-temporal coordinates (`self.x_data`, `self.y_data`, `self.t_data`) derived from the FDM snapshots are used as input. These are concatenated into a single `X_input` tensor of shape (6724, 3).
    2. **Network Prediction**: The neural network (e.g., `self.neural_network_model`) processes all 6724 points simultaneously in a single batch, outputting predicted `u` and `v` values (`u_pred_data`, `v_pred_data`) for each data point. The output tensor `uv` will have a shape of (6724, 2).
    3. **Error Calculation**: The squared difference between the network's predicted `u` and `v` values (`u_pred_data`, `v_pred_data`) and the corresponding ground truth `u` and `v` values (`self.u_data`, `self.v_data`) is calculated.
    4. **Mean Reduction**: `tf.reduce_mean()` is applied to these squared differences for both `u` and `v`, calculating the Mean Squared Error (MSE).
    5. **Summation**: The MSE for `u` and `v` are summed to form `loss_data`.
  * The `loss_data` quantifies how well the PINN's predictions align with the observed data.

* **2. Physics-Informed Loss (PDE Residual Loss, `loss_pde`)**:
  
  * **Purpose**: To enforce that the neural network's predictions satisfy the governing physical laws (2D Burgers' equations).
  * **Mechanism**:
    1. **Input Collocation Points**: The 60,000 fixed, randomly sampled spatio-temporal collocation points (`self.x_pde`, `self.y_pde`, `self.t_pde`) are used as input.
    2. **Network Prediction**: The neural network predicts `u` and `v` values (`u_pred_pde`, `v_pred_pde`) for all 60,000 collocation points in a single forward pass.
    3. **Automatic Differentiation**: Using TensorFlow's automatic differentiation, all necessary partial derivatives of `u_pred_pde` and `v_pred_pde` with respect to `x_pde`, `y_pde`, and `t_pde` are computed at each of the 60,000 points (e.g., $u_t, u_x, u_y, u_{xx}, u_{yy}, v_t, v_x, v_y, v_{xx}, v_{yy}$).
    4. **PDE Residual Calculation**: These predicted `u`, `v`, and their computed derivatives are then plugged into the 2D Burgers' equations, along with the discoverable kinematic viscosity `nu` (represented as `tf.exp(self.log_nu_pinn)`). This calculates the PDE residuals, $f_u$ and $f_v$, for each collocation point.
    5. **Loss Comparison**: The `loss_pde` component calculates the mean squared error of these residuals (`mean(f_u^2) + mean(f_v^2)`). The goal of this loss is to force the neural network's predictions to make the PDE residuals **zero**, thereby satisfying the governing physical laws.

* **Optimization Process**:
  
  * The optimization algorithms (Adam and L-BFGS-B) adjust the **internal parameters of the neural network** (its weights and biases) and the **discoverable physical parameter** (`log_nu_pinn`).
  * The **locations of the 60,000 collocation points are static inputs** and are **not adjusted** during training.

* **Training Phases**:
  
  * **"Data-Only Pre-training"**: An initial phase where **only the data fidelity loss** (using the 6724 points) is calculated and minimized. The PDE residual loss is not considered here.
  * **Full Loss Training**: After pre-training, the **full loss (combining both `loss_data` and `loss_pde`) is used** for subsequent Adam and L-BFGS-B optimizations. Both terms are calculated in every epoch/iteration during this phase.

### IV. Parameter Discovery with Real Datasets

When a real dataset is used to discover a PDE parameter (like `nu` in this case), the primary change is the **source of the `u_data` and `v_data`**.

* **Data Source**: Instead of FDM-generated ground truth, `u_data` and `v_data` would be loaded from real-world measurements (e.g., `.csv`, `.npz` files containing (x, y, t, u, v) tuples).
* **Data Preparation**: The data would still be prepared into appropriate tensor format, replacing the `generate_ground_truth_data` function with data loading and preprocessing steps.
* **PDE Collocation Points**: The generation of the 60,000 randomly sampled PDE points remains the same, as they are independent of the observed data and crucial for enforcing physical laws.
* **Loss Function**: The PINN's loss function still combines data fidelity and PDE residual loss, but the data loss now measures the discrepancy between predictions and **real observed data**.

**Key Considerations with Real Data**:

* **Noise**: Real data often contains noise, potentially requiring robust loss functions, regularization, or careful weighting of loss components.
* **Sparsity/Irregularity**: PINNs are well-suited for sparse or irregularly sampled real datasets, as they do not require a structured grid for training.
* **Uncertainty**: Advanced PINN approaches can incorporate inherent measurement uncertainty.
* **Domain Definition**: Spatial and temporal boundaries must accurately reflect the real data's domain.

### V. Comparison: PINN vs. Conventional Neural Network (Direct Problem)

For training a **conventional Neural Network (NN) for a direct problem** (e.g., predicting u, v given x, y, t):

* **Input and Output**: Similar to a PINN, it takes (x, y, t) as input and outputs (u, v).
* **Training Data**: A conventional NN relies **solely on a dataset** of input-output pairs (x_data, y_data, t_data, u_data, v_data), which must contain ground truth or observed values.
* **Loss Function**: The loss function **only consists of the data fidelity loss** (typically Mean Squared Error between network predictions and the dataset's ground truth). There is **no PDE residual loss component**.
* **Optimization**: The optimizer adjusts the NN's weights and biases to minimize this data-only loss.

**Key Differences from a PINN**:

* **No Physics Enforcement**: A conventional NN does not inherently ensure its predictions satisfy underlying physical laws. It only learns to fit the provided data, and its predictions might violate the PDE if data is sparse or noisy.
* **No Parameter Discovery**: Without the PDE residual, it cannot directly discover unknown parameters within the PDE.
* **Data Dependency**: Its performance and generalization capabilities are entirely dependent on the quantity, quality, and coverage of the training data. It cannot leverage physical laws to "fill in gaps".

In summary, a conventional NN is a pure data-driven model, while a PINN is a hybrid model that combines data-driven learning with physics-informed regularization.

<br><sub>Last edited: 2025-08-23 16:35:40</sub>
