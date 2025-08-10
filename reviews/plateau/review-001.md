# Review 001: Implementing ReduceLROnPlateau in main_plateau.py

## Review of Files

- `main_plateau.py`: Contains the PINN implementation for 2D Burgers' equation with a two-stage optimization process (Adam and L-BFGS-B). It already has variables and a partial logic for `ReduceLROnPlateau`.
- `curriculo-ponderado.md`: Describes "Weighted Curriculum" and "ReduceLROnPlateau" strategies.

## Main Conclusions from Original Code

The `main_plateau.py` script is designed for parametric PINN training and includes a section for `ReduceLROnPlateau` within its Adam optimization loop. However, the current implementation of `ReduceLROnPlateau` is incomplete and does not correctly update the TensorFlow variables for `best_loss`, `patience_counter`, and `learning_rate` using `tf.assign` operations. The instruction in `.gemini/GEMINI.md` explicitly states to *only* implement `ReduceLROnPlateau` and *not* the "Weighted Curriculum".

## Research on ReduceLROnPlateau

`ReduceLROnPlateau` is a common learning rate scheduling technique. It monitors a quantity (e.g., validation loss) and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced by a 'factor'. This helps the optimizer escape local minima and fine-tune the model. For custom training loops in TensorFlow 1.x (which this project uses by disabling eager execution), this logic needs to be implemented manually using `tf.assign` to update the TensorFlow variables.

## Next Steps Proposed based on Review

1. **Refine `ReduceLROnPlateau` logic in `PINN_Burgers2D.train`:**
   * Ensure `tf_v1.assign` is used to update `self.best_loss` and `self.patience_counter` correctly based on the current total loss.
   * Implement the actual learning rate reduction by using `tf_v1.assign` on `self.learning_rate` when the patience limit is reached.
   * Reset `patience_counter` to 0 after reducing the learning rate.
   * Add clear print statements to indicate when the learning rate is being reduced.
2. **Test with reduced epochs:** Temporarily set `adam_epochs_stage1` to a small value (e.g., 100) to quickly verify the `ReduceLROnPlateau` functionality.
3. **Run full training:** Revert `adam_epochs_stage1` to its original value (5000) for a full training run after successful verification.
4. **Update `FILES.md` and `SNAPSHOT.md`:** Document the changes made to `main_plateau.py` and the overall project state.

<br><sub>Last edited: 2025-08-22 06:29:38</sub>
