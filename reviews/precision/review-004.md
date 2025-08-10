# Review 004: First Experiments on Loss Function Weighting

## 1. Review of Files and Literature

-   **`src/main_precision.py`**: The code has been translated to English. The current loss function is hardcoded as `20 * loss_data + 1 * loss_pde`. This gives a high weight to the data-fitting term (`loss_data`) and a low weight to the physics-residual term (`loss_pde`). The number of Adam epochs is set to 1000.
-   **Literature (`cuomo2022.md`, etc.)**: The literature confirms that there is no single theory for setting these weights, but a common practice is to balance the terms so they are of a similar order of magnitude. The current weighting scheme in our code appears unbalanced.

## 2. Key Findings from Review

The primary finding is that our current loss function may be hindering the precision of the `nu` discovery by forcing the model to prioritize matching the data shape over respecting the physical laws where `nu` is a parameter. To improve `nu`'s accuracy, the model must be more sensitive to the PDE residual.

## 3. Proposed Next Steps

I will conduct a series of experiments to test the impact of the loss weights on the precision of the discovered `nu` parameter. To facilitate rapid testing, I will first modify the code.

**Step 3.1: Modify `main_precision.py` for Experimentation**

1.  **Reduce Adam Epochs**: I will change the number of Adam epochs from `1000` to `200`. This will significantly speed up each experimental run.
2.  **Parameterize Loss Weights**: I will modify the `compute_loss` function and the main execution block to easily configure the weights for `loss_data` and `loss_pde` without changing the core logic on every run.

**Step 3.2: Design and Execute Experiments**

I will run the script with four different weighting configurations to observe the effect on the final `nu` value. The true value of `nu` is `0.05`.

-   **Run 1 (Baseline):** `lambda_data = 20`, `lambda_pde = 1`
-   **Run 2 (Equal Weighting):** `lambda_data = 1`, `lambda_pde = 1`
-   **Run 3 (Prioritize PDE):** `lambda_data = 1`, `lambda_pde = 20`
-   **Run 4 (Strongly Prioritize PDE):** `lambda_data = 1`, `lambda_pde = 100`

For each run, I will capture the output in a dedicated log file and record the final discovered `nu` and its relative error.

<br><sub>Last edited: 2025-08-09 13:58:34</sub>
