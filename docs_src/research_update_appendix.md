# Appendix A: Methodological Advances - From Specialist to Generalist Surrogate

**Document Status:** Post-Submission Research Report (CIACA 2025)  
**Scope:** Advanced Computational Optimization and Sampling Strategies for PINNs

## A.1. Introduction

Following the acceptance of the original article "Parameter Discovery in the 2D Burgers' Equation", subsequent research focused on overcoming the **"instance-specific" limitation** identified in the conclusion. The research evolved from a **Specialist Model** (proof of concept, single $\nu$) to a **Generalist Surrogate Model** capable of parametric inference for $\nu \in [0.01, 0.1]$. This appendix details the computational optimizations and sampling strategies required to scale the solution, shifting the problem from "Controlled Overfitting" to "Efficient Generalization".

## A.2. Evolution of the Modeling Paradigm

The original approach required mandatory retraining for any new physical parameter. The new **Surrogate V2** architecture treats the kinematic viscosity $\nu$ not just as a coefficient in the PDE loss, but as an input feature to the neural network $f(x, y, t, \nu) \rightarrow (u, v)$, effectively learning the operator of the equation.

| Feature            | Specialist Model (Article)             | Surrogate Model (Current)            |
|:------------------ |:-------------------------------------- |:------------------------------------ |
| **Objective**      | Fit one specific instance ($\nu=0.05$) | Generalize for $\nu \in [0.01, 0.1]$ |
| **Training Data**  | Single Dataset                         | Multi-Dataset (19 simulations)       |
| **Inference Time** | ~520s (Retraining)                     | **< 60s (Direct Inference)**         |
| **Metric**         | Reconstruction Error (Training)        | Generalization Error (Unseen Data)   |

## A.3. Computational Optimization for Scale

Scaling to multi-dataset training introduced severe memory bottlenecks on the reference hardware (NVIDIA RTX 3050, 6GB VRAM).

### A.3.1. Memory-Bound Regime and Internal Batching

Profiling with **NVIDIA Nsight Compute** (`ncu`) revealed that the training process was **Memory-Bound**, dominated by data transfer latency rather than arithmetic intensity. The predominance of element-wise operations on large batches saturated the memory bandwidth.

* **Optimization:** We implemented *internal batching* for the PDE residual evaluation, reducing the `pde_batch_size` from 20,000 to **4,096**.
* **Result:** This yielded measurable gains:
  * **L2 Cache Hit Rate:** Increased by **~16%**.
  * **Global Memory Traffic:** Reduced by **~67%** (mitigating cache thrashing).
  * **Training Time:** Estimated **~2x reduction** in Stage 1 training time.

### A.3.2. Nested Gradient Tapes for Stability

Calculating second-order derivatives ($u_{xx}, u_{yy}$) for the Navier-Stokes/Burgers residual is computationally expensive. The standard implementation using persistent `tf.GradientTape` caused systematic **Out-Of-Memory (OOM)** errors due to excessive retention of the computational graph.

* **Solution:** We implemented **Nested Gradient Tapes** with explicit resource release. This reduced the spatial complexity of the automatic differentiation graph, preventing VRAM saturation during the backward pass.

## A.4. Sampling Strategies and Robustness

### A.4.1. The Failure of Random Sampling

Initial experiments with random sampling (Ensemble of 10 seeds) for the Surrogate V2 model showed high instability. While the best case (Seed 7) achieved a generalization error of **3.54%**, the worst cases diverged to **162%**, with a high standard deviation ($\sigma \approx 57.87\%$).

* **Failure Analysis:** The optimizer in Stage 2 (Inverse Problem) was often "deceived" by the neural network. The loss surface learned by the surrogate contained spurious gradients that pushed the solution to the physical domain boundary ($\nu \approx 0.1$) when trained on sparse data.

### A.4.2. Stabilization via Latin Hypercube Sampling (LHS)

To ensure uniform coverage of the parameter space $\nu$, we replaced random sampling with **Latin Hypercube Sampling (LHS)**.

* **Method:** LHS stratifies the input probability distributions, ensuring that each portion of the parameter range is sampled exactly once.
* **Results (Ensemble Statistics):**
  * **Mean Error:** Reduced from 74.07% to **20.56%** (~3.6x improvement).
  * **Stability:** Standard deviation reduced by ~2.7x (from 57.87% to 21.46%).
  * **Critical Failures:** LHS eliminated the divergence scenarios where error exceeded 100%.

### A.4.3. Focused Optimization (Best Case Refinement)

To test the limit of the architecture, we extended the training of the best LHS candidate (Seed 2) from 6,000 to **15,000 epochs**.

* **Result:** The generalization error dropped from 2.63% to **2.53%** (-3.61% relative improvement). This confirms that the surrogate model quality is the limiting factor for inference accuracy.

## A.5. Conclusion: The Virtual Sensor

The research successfully transformed the PINN from a single-instance solver into a **Virtual Sensor**. Although the initial training cost is higher (~19 min vs ~8.7 min for the Specialist), the surrogate model enables **instantaneous inference** ($< 47s$) of physical parameters for new, unseen flows without retraining. The combination of **Memory-Bound optimizations** and **LHS sampling** was decisive in making this approach viable on constrained hardware.

## A.6. References

1. **Abadi, M. et al. (2015).** TensorFlow: Large-scale machine learning on heterogeneous distributed systems. *arXiv preprint arXiv:1603.04467*.
2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
3. **Wang, S., Teng, Y., & Perdikaris, P. (2021).** Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*.
4. **McKay, M. D. et al. (1979).** Comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*. (Context: Latin Hypercube Sampling).
