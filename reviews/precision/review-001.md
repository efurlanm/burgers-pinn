# Review 001: Initial Analysis and Plan Forward

## File Review

- **`GEMINI.md`**: The core instructions are clear. The main goal is to update `manuscript.tex` to focus on the "precision of parameter discovery" for the 2D Burgers' equation using PINNs. This involves creating a new Python script dedicated to this task, while maintaining the existing code as an alternative approach. All code and documentation must be in English.

- **`manuscript.tex`**: The manuscript is well-structured but requires significant updates. The introduction and related work sections are solid, but the methodology, results, and future work sections will need to be heavily revised to reflect the new focus on parameter discovery precision. The bibliography is in place, but some formatting and content issues need to be addressed.

- **`src/main_scipy.py`**: This script uses a hybrid TensorFlow 1.x and SciPy approach. It implements a PINN to solve the inverse problem for the 2D Burgers' equation. It includes a two-stage optimization (Adam then L-BFGS-B) and generates its own training data using an FDM simulation. The code is functional but needs to be translated to English and better commented. A key feature is its use of multiple time steps for training data, which is crucial for parameter identifiability.

- **`src/main_tfp.py`**: This script uses TensorFlow 2.x and TensorFlow Probability. It is more modern than the SciPy version but trains on a single time step, which, as noted in the manuscript, can be problematic for parameter discovery. This script will be kept as a reference for the alternative methodology.

- **`src/finite_difference.py`**: A standalone FDM solver for the 2D Burgers' equation. It is written in Portuguese and is not directly integrated with the main PINN scripts. It serves as a good reference for the FDM implementation.

- **`src/plot_results.py`**: A simple script for visualizing the results from the `.npz` files. It is also in Portuguese and will need to be translated.

- **`reviews/`**: The existing review files in `lbfgsb-scipy` and `lbfgsb-tfp` provide a good history of the project's development. The new reviews for the precision focus will be created in `reviews/precision/`.

## Key Findings from the Original Code

1.  **Two Implementations**: The project has two distinct PINN implementations: one using a hybrid TF1.x/SciPy approach and another using TF2.x/TFP.
2.  **Parameter Identifiability**: The `main_scipy.py` script uses multiple time steps for training, which is a key technique for improving parameter identifiability, as mentioned in the manuscript. The `main_tfp.py` script does not, which makes it a good example of the problem.
3.  **Language and Comments**: The code and comments are a mix of English and Portuguese. This needs to be standardized to English.
4.  **Standalone Scripts**: The FDM and plotting scripts are standalone and not integrated into a cohesive workflow.

## Proposed Next Steps

1.  **Create `main_precision.py`**: I will create a new script, `src/main_precision.py`, which will be based on `src/main_scipy.py` due to its more robust multi-time-step approach. This new script will be cleaned up, translated to English, and heavily commented. It will be the primary tool for the new research focus.

2.  **Translate and Refactor**: I will translate all comments, variable names, and output strings in the new `src/main_precision.py` script to English. I will also refactor the code for clarity and maintainability.

3.  **Update `manuscript.tex`**: I will begin updating the manuscript to reflect the new focus. This will include:
    -   Rewriting the methodology section to describe the new `main_precision.py` script.
    -   Updating the results section with the new findings from the precision-focused experiments.
    -   Revising the future work section to align with the new research direction.

4.  **Create `FILES.md`**: I will create a `FILES.md` file to document the project structure and the purpose of each file, as instructed.

<br><sub>Last edited: 2025-08-09 11:32:11</sub>
