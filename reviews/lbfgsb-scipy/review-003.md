# Review 003: Comparison of L-BFGS-B Implementations

## Precise and Detailed Analysis

### Comparison of L-BFGS-B in `burgers2d-03.py` and `l-bfgs-b-tf1.md`

**`l-bfgs-b-tf1.md` describes L-BFGS-B usage in TensorFlow v1.x via `tf.contrib.opt.ScipyOptimizerInterface`.**

- **Key takeaway from `l-bfgs-b-tf1.md`:** This document emphasizes that for TensorFlow v1.x, L-BFGS-B is typically accessed through `tf.contrib.opt.ScipyOptimizerInterface`, which leverages SciPy's optimization algorithms. It also explicitly states: "For TensorFlow 2.x, the recommended way to use L-BFGS-B is through the TensorFlow Probability (TFP) library, specifically `tfp.optimizer.lbfgs_minimize`."
- **Observed Convergence in `l-bfgs-b-tf1.md`:** The document highlights very good convergence for a Burgers PINN training, with 6880 iterations, and precise identification of parameters with low errors.

**`burgers2d-03.py` implements L-BFGS-B using `scipy.optimize.minimize`.**

- **Implementation in `burgers2d-03.py`:** The current code uses `scipy.optimize.minimize` directly, passing a function `loss_and_grads` that returns both the loss and the flattened gradients. This is a standard way to use SciPy's L-BFGS-B.
- **Optimizer Options in `burgers2d-03.py`:**
    - `maxiter`: 50000
    - `maxfun`: 50000
    - `maxcor`: 50
    - `maxls`: 50
    - `ftol`: `1.0 * np.finfo(float).eps` (machine epsilon for float).

### Discrepancies and Potential Issues for Convergence Precision

1.  **TensorFlow Version Mismatch:** The most significant discrepancy is that `l-bfgs-b-tf1.md` explicitly recommends `tfp.optimizer.lbfgs_minimize` for TensorFlow 2.x, while `burgers2d-03.py` uses `scipy.optimize.minimize`. While `scipy.optimize.minimize` can work with TensorFlow tensors (by converting them to NumPy arrays for the `fun` and `jac` arguments), using `tfp.optimizer.lbfgs_minimize` might offer better integration with TensorFlow's automatic differentiation and graph mode execution, potentially leading to more stable and precise convergence.

2.  **`ftol` Setting:** The `ftol` (function value tolerance) is set to `1.0 * np.finfo(float).eps`. This is an extremely strict tolerance. While it aims for high precision, it might also cause the optimizer to struggle to converge or to take an excessive number of iterations if the numerical precision of the loss function or gradients is not high enough, or if the problem is ill-conditioned. The `l-bfgs-b-tf1.md` example also uses this `ftol`, but it's worth noting as a potential factor in convergence issues.

3.  **Gradient Calculation:** `burgers2d-03.py` calculates gradients using `tf.GradientTape()` and then flattens them. This is correct for `scipy.optimize.minimize`. However, `tfp.optimizer.lbfgs_minimize` is designed to work directly with TensorFlow tensors and `tf.GradientTape`, potentially streamlining the process and avoiding intermediate NumPy conversions.

4.  **Adam Pre-training:** `burgers2d-03.py` uses a substantial Adam pre-training phase (40,000 epochs). This is a common practice to get the model into a good region of the loss landscape before applying L-BFGS-B. The `l-bfgs-b-tf1.md` example mentions 0 Adam iterations for noiseless data, implying L-BFGS-B was the primary optimizer. The effectiveness of the Adam pre-training can significantly impact the starting point and subsequent convergence of L-BFGS-B.

## Proposed Next Steps based on Review and Research

1.  **Investigate `tfp.optimizer.lbfgs_minimize`:** The most promising avenue for improving convergence precision and better aligning with TensorFlow 2.x best practices is to switch from `scipy.optimize.minimize` to `tfp.optimizer.lbfgs_minimize`. This would involve:
    -   Importing `tensorflow_probability as tfp`.
    -   Modifying the `train` method to use `tfp.optimizer.lbfgs_minimize`.
    -   Adapting the `loss_and_grads` function (or its equivalent) to work directly with TFP's requirements, which typically involve returning the loss and a list of gradients for trainable variables.
2.  **Evaluate `ftol`:** While `1.0 * np.finfo(float).eps` is a common setting for high precision, it might be too aggressive. Consider experimenting with a slightly larger `ftol` (e.g., `1e-9` or `1e-12`) if the switch to TFP doesn't immediately resolve convergence issues. However, given the context of the paper, the goal is high precision, so this should be a last resort.
3.  **Monitor Convergence:** Implement more detailed logging during the L-BFGS-B phase to track the loss, gradients, and discovered `nu` at each iteration. This will help in diagnosing convergence behavior.
4.  **Document Changes:** Record all changes made and their impact on the convergence precision in subsequent review files.

I will now proceed with the first proposed next step: investigating and implementing `tfp.optimizer.lbfgs_minimize` in `burgers2d-03.py`.
<br><sub>Last edited: 2025-08-02 22:47:39</sub>
