## The best approach

Rather than seeking a completely different method, the most effective approach is to:

- Implement the two-stage Parametric PINN: This is the foundation that offers the greatest performance gain with a reasonable implementation effort.

- If necessary, add Adaptive Sampling: This is the most logical subsequent improvement. It is a proven method for increasing accuracy on problems with steep gradients, and its implementation is a modification of the sampling process, not a complete redesign of the network.

A scientific paper that represents the most promising and practical next step for your project would be one that details Adaptive Sampling:

- Title: Adaptive causal sampling for physics-informed neural networks

- Authors: Guo, J., Wang, H., & Hou, C. (2022)

- Why it's relevant: This paper offers a strategy to improve Parametric PINN training by focusing computational resources on areas of the domain where error is highest, and is a natural evolution of its current implementation.



## Analysis and Implementation of a Parametric Physics-Informed Neural Network for Generalization in the 2D Burgers' Equation Inverse Problem

The generalization of Physics-Informed Neural Networks (PINNs) across a spectrum of physical parameters, such as viscosity (nu) in the 2D Burgers' equation, is a critical challenge that limits their utility in inverse problems. A standard PINN, optimized for a single parameter instance, is computationally inefficient for inverse problem-solving, as it necessitates complete retraining for each new parameter evaluation. A scientifically robust method to overcome this limitation is the **Parametric PINN**. This approach reframes the learning objective from approximating a single solution to learning a surrogate model that maps both spatio-temporal coordinates and the physical parameter to the solution field. This analysis confirms that a parametric approach is a promising and direct method for implementation within the existing `main_precision.py` codebase , with further enhancements available through adaptive training strategies.

---

#### 1. Viability and Scientific Foundation of the Parametric PINN Approach

The Parametric PINN approach is academically sound and represents the standard methodology for creating generalizable surrogate models. The central concept is to treat the viscosity parameter, nu, as an additional input to the neural network. Consequently, the network learns to approximate the solution function $u(x, y, t, \nu)$, effectively capturing the solution's dependency on viscosity.

This method is extensively supported by scientific literature, which identifies the single-instance training of PINNs as a primary limitation for applications in design optimization, uncertainty quantification, and inverse problems.2 By training a single model over a predefined range of $\nu$ values (e.g., nuin[nu_min,nu_max]), the resulting network acts as a rapid and generalizable solver, capable of inferring solutions for any $\nu$ within that range without retraining. This capability is essential for an efficient inverse problem solver.

---

#### 2. Step-by-Step Implementation Guide for `main_precision.py`

Implementing the Parametric PINN is a direct refactoring process. The objective is to transition the model from *discovering* a single $\nu$ to *generalizing* over a range of $\nu$ values.

##### Step 1: Modify the Network Architecture to Accept the New Parameter

The network must be adapted to accept four inputs: `x`, `y`, `t`, and $\nu$. This requires changing the dimension of the first layer in the network definition.

- In `main_precision.py` :
  
  Python
  
  ```
  # --- Neural Network Architecture ---
  # Original: Input: (x, y, t), Output: (u, v)
  layers =
  ```

- Modified Code:
  
  Python
  
  ```
  # --- Neural Network Architecture ---
  # New: Input: (x, y, t, nu), Output: (u, v)
  layers =
  ```

##### Step 2: Update the Forward Pass to Process the Viscosity Input

The `neural_network_model` function must be updated to handle the four-dimensional input tensor. This includes scaling (normalizing) the $\nu$ input, which is critical for stable training.

- In `PINN_Burgers2D` class :
  
  Python
  
  ```
  def neural_network_model(self, X):
      #... (original implementation)
      H = tf.concat([x_scaled, y_scaled, t_scaled], axis=1)
      #... (forward pass logic)
      return Y
  ```

- Modified Code:
  
  Python
  
  ```
  def neural_network_model(self, X, nu_min, nu_max): # Add nu bounds for scaling
      """    Defines the forward pass of the parametric neural network.    """
      # Input scaling to the range [-1, 1]
      x_scaled = 2.0 * (X[:, 0:1] - self.x_min) / (self.x_max - self.x_min) - 1.0
      y_scaled = 2.0 * (X[:, 1:2] - self.y_min) / (self.y_max - self.y_min) - 1.0
      t_scaled = 2.0 * (X[:, 2:3] - self.t_min) / (self.t_max - self.t_min) - 1.0
      # Scale the new $\nu$ input
      nu_scaled = 2.0 * (X[:, 3:4] - nu_min) / (nu_max - nu_min) - 1.0
  
      H = tf.concat([x_scaled, y_scaled, t_scaled, nu_scaled], axis=1)
  
      # Forward pass logic remains the same
      for l in range(len(self.weights) - 1):
          W, b = self.weights[l], self.biases[l]
          H = tf.tanh(tf.add(tf.matmul(H, W), b))
      W, b = self.weights[-1], self.biases[-1]
      Y = tf.add(tf.matmul(H, W), b)
      return Y
  ```

##### Step 3: Decouple Viscosity from Trainable Model Variables

The current implementation treats `log_nu_pinn` as a trainable variable. This must be removed. The viscosity will now be a data input provided during training.

- In `PINN_Burgers2D.__init__` :
  
  Python
  
  ```
  # Remove this section
  # self.log_nu_pinn = tf.Variable(tf.math.log(0.06), dtype=tf.float32, name="log_nu_pinn")
  # self.trainable_variables = self.weights + self.biases + [self.log_nu_pinn]
  ```

- Modified Code:
  
  Python
  
  ```
  # The trainable variables are now only the network's weights and biases
  self.trainable_variables = self.weights + self.biases
  ```

##### Step 4: Adapt the PDE Residual and Loss Functions

The `compute_pde_residual` function must be modified to accept $\nu$ as an argument instead of calculating it from `self.log_nu_pinn`.

- In `PINN_Burgers2D` class :
  
  Python
  
  ```
  def compute_pde_residual(self, x, y, t):
      #... (derivative calculations)...
      nu = tf.exp(self.log_nu_pinn)
      f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
      #...
  ```

- Modified Code:
  
  Python
  
  ```
  # The function now accepts $\nu$ as an input tensor
  def compute_pde_residual(self, x, y, t, nu):
      # predict_velocity must also be updated to handle the 4th input
      u, v = self.predict_velocity(x, y, t, nu)
  
      #... (derivative calculations remain the same)...
  
      # The PDE residual now uses the input nu
      f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
      f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
      return f_u, f_v
  ```
  
  The `predict_velocity` and `compute_loss` functions must also be updated to handle the new $\nu$ input and pass it down accordingly.

##### Step 5: Modify the Training Data Generation

During training, for each batch of collocation points `(x_pde, y_pde, t_pde)`, a corresponding batch of `nu_pde` values must now also be provided, sampled randomly from the desired training range.

- Conceptual Modification (to be placed in the training loop):
  
  Python
  
  ```
  # Define the range for viscosity training
  nu_min_train, nu_max_train = 0.01, 0.1
  
  # Inside the training loop (e.g., train_step_adam), generate these values for each batch.
  # This replaces the static generation of collocation points.
  x_pde_batch = np.random.uniform(x_min, x_max, (batch_size, 1))
  y_pde_batch = np.random.uniform(y_min, y_max, (batch_size, 1))
  t_pde_batch = np.random.uniform(t_min, t_max, (batch_size, 1))
  nu_pde_batch = np.random.uniform(nu_min_train, nu_max_train, (batch_size, 1))
  
  # This batch of $\nu$ values must be fed into the loss function along with x, y, and t.
  feed_dict = {
      self.x_pde_placeholder: x_pde_batch,
      self.y_pde_placeholder: y_pde_batch,
      self.t_pde_placeholder: t_pde_batch,
      self.nu_pde_placeholder: nu_pde_batch,
      #... other placeholders for data points...
  }
  self.session.run(self.train_op_adam, feed_dict)
  ```

---

#### 3. Optimal Solution and Strategic Recommendations

A strategic path forward involves incremental enhancements to address the generalization challenge effectively.

##### Primary Path: Parametric PINN Implementation

The first and most critical step is the implementation of the **Parametric PINN** as detailed above. This modification directly addresses the core problem by enabling the model to learn the solution's dependency on the viscosity parameter. It provides the best balance of implementation effort and performance gain for achieving generalization.

##### Secondary Enhancement: Adaptive Sampling

After implementing the parametric model, if convergence issues or inaccuracies persist, especially in low-viscosity regimes where sharp gradients (shocks) form, the next logical step is to implement **Adaptive Sampling**. This is a training strategy, not an architectural change, making it easier to implement than a full reformulation like FO-PINN.

- **Concept:** Instead of sampling collocation points uniformly, adaptive methods periodically evaluate the PDE residual across the domain and add new training points in regions where the error is highest. This forces the network to focus its capacity on the most challenging parts of the solution space, such as shock fronts.

- **Advantage:** This technique directly targets the regions where the model is failing, improving accuracy without altering the network architecture. It is a well-documented method for improving PINN robustness.

##### Advanced Alternative: First-Order PINN (FO-PINN)

If the combination of a parametric model and adaptive sampling is still insufficient, a more fundamental change can be considered. The **FO-PINN** approach addresses numerical instabilities associated with high-order derivatives, which are often a root cause of failure in complex parameterized problems.

- **Concept:** The second-order Burgers' equation is mathematically reformulated into an equivalent system of first-order equations by introducing auxiliary variables for the first derivatives (e.g., p=partialu/partialx).

- **Advantage:** This method has been shown to yield significantly higher accuracy for parameterized systems by smoothing the loss landscape and making the optimization task more stable. It can also accelerate training by removing the need for computationally expensive second-order automatic differentiation.

**Recommendation Summary:**

1. **Primary Path:** Implement the **Parametric PINN** as detailed in the step-by-step guide. This is the most direct and scientifically sound path to achieving the generalization goal.

2. **Secondary Enhancement:** If needed, augment the parametric model with an **Adaptive Sampling** strategy during training to improve accuracy in complex, high-gradient regions.

3. **Advanced Alternative:** For maximum accuracy and stability, consider refactoring the problem using the **FO-PINN** formulation.

### Detailed Analysis of the Parametric PINN and its Implementation

#### 1. The Conceptual Shift: From Parameter Discovery to Generalization

To understand the required changes, it is crucial to differentiate the objective of the current code (`main_precision.py`) from the objective of the proposed model.

- **Current Model (Parameter Discovery):** The `main_precision.py` script implements a PINN for an **inverse problem**. Its goal is to discover a **single value** for the viscosity, $\nu$. To achieve this, `self.log_nu_pinn` is defined as a trainable variable (`tf.Variable`) and included in the `self.trainable_variables` list. During optimization, the algorithm simultaneously adjusts the network weights (`weights` and `biases`) and the value of `log_nu_pinn` to minimize the loss function. Ultimately, the model converges to the $\nu$ value that best explains the provided training data. **In this case, viscosity is a model parameter to be learned.**

- **Proposed Model (Parameter Generalization):** The Parametric PINN approach transforms the model into a **generalizable solver** or **surrogate model**. The objective is no longer to discover a specific $\nu$, but rather to teach the network to understand how the solution `(u, v)` behaves for **any** value of $\nu$ within a predefined range (e.g., nuin[0.01,0.1]).

To directly answer the question: **No, with the new approach, the viscosity parameter $\nu$ is no longer a variable trained alongside the weights and biases.** It becomes a **conditional input** to the network. The network learns the mapping:

$(u, v) = \mathcal{NN}(x, y, t, \nu; \theta)$

Where theta represents only the weights and biases, which are the sole trainable parameters.

#### 2. How it Works: The Training and Inference Process

1. **Training:**
   
   - At each training step, a batch of collocation points `(x, y, t)` is sampled from the domain, as before.
   
   - The fundamental difference is that for this same batch, a corresponding set of $\nu$ values is also randomly sampled from the training range (e.g., `np.random.uniform(0.01, 0.1, batch_size)`).
   
   - The neural network receives the `(x, y, t, nu)` set as input.
   
   - The PDE residual is calculated using the specific $\nu$ value for each collocation point in that batch.
   
   - The optimizer adjusts the weights and biases (theta) to minimize the residual error across the entire domain and for the full range of sampled viscosities. The model is forced to learn the functional relationship between viscosity and the solution dynamics.

2. **Inference (Post-training):**
   
   - Once trained, the model can be used as a rapid solver.
   
   - To obtain the solution for a specific viscosity not seen during training (e.g., `nu = 0.035`), one simply provides this value as the fourth input to the network, along with the desired `(x, y, t)` coordinates. The network will predict the `(u, v)` solution corresponding to that physics without requiring any retraining.

#### 3. How Does This Affect `plot_main_figure.txt`?

The file `plot_main_figure.txt` was not provided, but it can be inferred that it uses the results saved by `main_precision.py` to generate plots. The changes to how these results are generated are significant:

- **Current Data Generation:** The current script, upon completion, saves the predicted solution (`u_pinn_pred`, `v_pinn_pred`) that corresponds to the **single value of $\nu$ that was discovered** during training. It can only generate the solution for this specific viscosity.

- **New Data Generation Capability:** With the trained parametric model, the process of generating data for plotting becomes much more flexible. It will be possible to generate and save solutions for **multiple values of $\nu$** for comparison. For example, the prediction code could be structured as follows:
  
  Python
  
  ```
  # Viscosities to analyze
  nu_values_for_plotting = [0.01, 0.05, 0.1]
  
  # Coordinates for the final time slice
  X_plot_flat = X_np.flatten()[:, None]
  Y_plot_flat = Y_np.flatten()[:, None]
  T_plot_flat = np.full_like(X_plot_flat, t_max)
  
  # Dictionary to store results
  results_for_plotting = {}
  
  for nu_val in nu_values_for_plotting:
      # Create the $\nu$ input tensor, with the same shape as the coordinate tensors
      Nu_plot_flat = np.full_like(X_plot_flat, nu_val)
  
      # Predict the solution for this specific nu
      u_pred, v_pred = pinn.session.run(
          pinn.predict_velocity(
              tf.constant(X_plot_flat, dtype=tf.float32),
              tf.constant(Y_plot_flat, dtype=tf.float32),
              tf.constant(T_plot_flat, dtype=tf.float32),
              tf.constant(Nu_plot_flat, dtype=tf.float32) # Provide the new $\nu$ input
          )
      )
  
      # Store the result
      results_for_plotting[f'u_pred_nu_{nu_val}'] = u_pred.reshape((grid_points_y, grid_points_x))
      results_for_plotting[f'v_pred_nu_{nu_val}'] = v_pred.reshape((grid_points_y, grid_points_x))
  
  # Save the 'results_for_plotting' dictionary to a.npz file
  # The plotting script can then load these multiple solutions and compare them.
  ```

In summary, the fundamental change is that the plotting script will no longer be limited to visualizing a single solution. It will be able to load and compare a range of solutions for different viscosities, all generated by a **single trained model**. This enables a much richer analysis of the system's behavior and the model's generalization quality.

### How the Parametric PINN Solves the Inverse Problem (Parameter Discovery)

The core idea is to decouple the complex task of learning the PDE's solution space from the simpler task of finding a single parameter. This is achieved through a two-stage process.

#### Stage 1: Training a Generalizable Surrogate Model (The "Parametric PINN")

Before tackling the inverse problem, the first step is to train a single, powerful PINN that understands the physics of the Burgers' equation across a *range* of possible viscosity values. This is the model depicted on the right side of the Figure:

![parametric](img/parametric-pinn.png)

*Stage 1 - comparison between vanilla PINN and the new proposed parametric PINN*

- **Objective:** The goal of this stage is **not** to find the specific $\nu$ for the problem. Instead, the objective is to create a fast and accurate surrogate model, $\mathcal{NN}(x, y, t, \nu; \theta)$, that can predict the velocity field `(u, v)` for **any given viscosity $\nu$** within a predefined range (e.g., $\nu \in [0.01,0.1]$).

- **How it Works:**
  
  1. **Conditional Input:** As shown in the figure, the viscosity $\nu$ is treated as a fourth input to the network, alongside `x`, `y`, and `t`.
  
  2. **Training Process:** During training, the model is fed batches of random `(x, y, t)` points. For each point, a $\nu$ value is also randomly sampled from the training range. The loss function then forces the network's output to satisfy the Burgers' equation for that specific $\nu$.
  
  3. **Learned Parameters:** The only parameters being trained (optimized) are the network's **weights** and **biases**, denoted by $\theta$. The viscosity $\nu$ is simply input data.

- **Outcome:** The result of this stage is a highly valuable, reusable asset: a single trained neural network that acts as a general-purpose solver for the Burgers' equation. It has learned the relationship between the viscosity and the resulting flow dynamics.

#### Stage 2: Solving the Inverse Problem (Discovering the Unknown $\nu$)

Now, with the trained **surrogate model** in hand, the inverse problem becomes a much simpler and faster optimization task.

- **Objective:** The goal is to find the single, specific value of $\nu$ that causes the predictions of the surrogate model to best match the observed, real-world data points ($u_{data}$,  $v_{data}$).

- **How it Works:**
  
  1. **Freeze the Network:** The weights and biases ($\theta$) of the trained Parametric PINN are **frozen**. They are no longer trainable. The complex PDE-solving part is considered done.
  
  2. **Define a New, Simpler Optimization:** A new, lightweight optimization problem is set up where the **only trainable variable is $\nu$**.
  
  3. New Loss Function: The loss function for this stage is simply the Mean Squared Error (MSE) between the model's predictions and the observed data:
     
     Loss_inverse = MSE( $u_{predicted}$, $u_{data}$ ) + MSE( $v_{predicted}$, $v_{data}$ )
     
     where $u_{predicted}, v_{predicted} = Surrogate Model (x_{data}, y_{data}, t_{data}, \nu)$.
  
  4. **Optimization Loop: A standard optimizer (like L-BFGS or Adam) is used to find the value of $\nu$ that minimizes this `Loss_inverse`**. The process is as follows:
     
     - Start with an initial guess for $\nu$ (e.g., `nu_guess = 0.05`).
     
     - Use the pre-trained surrogate model to quickly predict the solution field for `nu_guess`.
     
     - Calculate the `Loss_inverse` by comparing this prediction to the real data.
     
     - The optimizer uses the gradient of this loss with respect to `nu_guess` to propose a better value (e.g., `nu_guess = 0.048`).
     
     - Repeat until the loss is minimized. The final `nu_guess` is the discovered parameter.

This process is extremely fast because each step only involves a forward pass through the already-trained network and a simple gradient calculation with respect to a single scalar variable ($\nu$), not the millions of weights in the network.

The key benefit is that if you have a *new* dataset corresponding to a *different* unknown viscosity, you do **not** need to repeat the long Stage 1 training. You can immediately reuse the same surrogate model and run the quick Stage 2 optimization to find the new $\nu$.

##### Stage 2 - details

This section highlights the most crucial and advantageous difference in the parametric approach. "Stage 2" is not equivalent to the vanilla PINN method for solving the inverse problem.

While in both cases ν is an optimized variable, the context and process are fundamentally different and significantly more efficient in Stage 2. The primary distinction is what else is being trained concurrently.

Here is a direct comparison:

---

###### Vanilla PINN (The current method in `main_precision.py`)

- **Trainable Variables** $(\theta)$ : The network's weights, biases, **AND** the viscosity (ν) are all adjusted simultaneously.

- **Optimization Objective**: The optimizer must perform two complex tasks at once:
  
  1. Adjust millions of weights and biases for the network to learn the solution to the partial differential equation (the system's physics).
  
  2. Adjust the single parameter ν so that the learned solution matches the provided observational data.

- **Computational Cost**: Very high. In each step, gradients are computed for all variables (weights, biases, and ν), and the entire network is updated. The model is learning the physics from scratch while also searching for ν.

###### Stage 2: Parameter Discovery (Using the Pre-Trained Parametric PINN)

- **Trainable Variables** ($\theta$) :  **ONLY** the viscosity (ν).

- **Network State**: The *weights* and *biases* of the neural network are **frozen** (non-trainable). The network has already learned the physics of the problem during *Stage 1*.

- **Optimization Objective**: The optimizer has a single, much simpler task:
  
  1. Adjust the **single parameter ν** to minimize the difference (Mean Squared Error) between the pre-trained surrogate model's predictions and the observed data.

- **Computational Cost**: Extremely low. The process only requires a forward pass of the data through the network and the calculation of the gradient with respect to a single scalar variable, ν. No backpropagation is needed to update network weights.

---

###### Analogy

- The **Vanilla PINN** approach is analogous to designing and building an engine from scratch and, in the same process, trying to guess the optimal type of fuel it should use.

- The **Two-Stage Parametric Approach** is different:
  
  - **Stage 1**: A highly versatile engine is first built to operate efficiently across a wide spectrum of fuels (this is the training of the parametric surrogate model). This is a heavy-duty task performed only once.
  
  - **Stage 2**: Given a sample of an unknown fuel (the observational data), the engine is not rebuilt. Instead, a single knob on the pre-built engine is turned (optimizing only ν) until it runs perfectly with that specific fuel. This is a rapid and efficient tuning process.

In summary, during Stage 2, the neural network is not being trained. It is being used as a highly efficient calculator—a surrogate model—to find the value of ν that best fits the observed data.

## References

Hassanaly, M., Weddle, P. J., King, R. N., De, S., Doostan, A., Randall, C. R., Dufek, E. J., Colclasure, A. M., & Smith, K. (2024). *PINN surrogate of Li-ion battery models for parameter inference. Part II: Regularization and application of the pseudo-2D model* (No. arXiv:2312.17336). arXiv. [[2312.17336] PINN surrogate of Li-ion battery models for parameter inference. Part II: Regularization and application of the pseudo-2D model](https://doi.org/10.48550/arXiv.2312.17336)

Daw, A., Bu, J., Wang, S., Perdikaris, P., & Karpatne, A. (2022). Rethinking the importance of sampling in physics-informed neural networks. *arXiv Preprint arXiv:2207.02338*.

Gladstone, R. J., Nabian, M. A., & Meidani, H. (2022). FO-pinns: A first-order formulation for physics informed neural networks. *arXiv Preprint arXiv:2210.14320*.

Guo, J., Wang, H., & Hou, C. (2022). Adaptive causal sampling for physics-informed neural networks. *arXiv Preprint arXiv:2210.12914*.

Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, *63*(1), 208–228.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, *378*, 686–707.

Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, *43*(5), A3055–A3081.

<br><sub>Last edited: 2025-08-18 13:36:58</sub>
