# Review 003: Implementing Ground Truth Collocation Points for Data and PDE Loss

## Objective

The main objective of this experiment is to modify the `main_coloc.py` script to use the "Ground Truth 41x41 points" (totaling 6724 points) for *both* the data fidelity loss and the PDE residual loss. This ensures that both terms of the loss function are evaluated at the same spatial and temporal points, as specified in the `GEMINI.md` objective.

## Current Status

`review-002.md` has been updated with the results from the previous experiment, where $\lambda_{pde} = 100.0$ was used with the original 60000 PDE collocation points.

## Plan

1.  **Read `main_coloc.py`**: Understand the current implementation of data loading and PDE collocation point generation.
2.  **Identify Data and PDE Point Generation**: Locate where `x_data_tf`, `y_data_tf`, `t_data_tf` (for data loss) and `x_pde`, `y_pde`, `t_pde` (for PDE loss) are defined.
3.  **Modify PDE Point Assignment**: Change the assignment of `x_pde`, `y_pde`, `t_pde` to directly use `x_data_tf`, `y_data_tf`, `t_data_tf` respectively. This will ensure that the same 6724 ground truth points are used for both terms.
4.  **Reduce Epochs for Testing**: Temporarily reduce the number of Adam and Data-Only pre-training epochs to a small value (e.g., 100 for Adam, 1000 for Data-Only) to quickly verify that the code runs without errors after the modification.
5.  **Run Test Experiment**: Execute the modified `main_coloc.py` with the reduced epochs.
6.  **Verify Changes**: Check the log output to confirm that the PDE loss is being calculated using the correct number of points (6724) and that the program runs without errors.
7.  **Revert Epochs**: Restore the original number of epochs (2000 for Adam, 10000 for Data-Only) for full training.
8.  **Update `FILES.md`**: Add `review-003.md` to the `FILES.md` document.

## Next Steps

After successfully implementing and verifying the change, a new experiment will be conducted with the full number of epochs and the updated collocation point strategy. The results will be documented in a subsequent review file.
<br><sub>Last edited: 2025-08-24 09:13:42</sub>
