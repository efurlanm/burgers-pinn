# Review 001: Initial Parametric PINN Experiment

## Objective

The primary objective of this experiment was to implement a Parametric Physics-Informed Neural Network (PINN) for the 2D Burgers' equation, capable of generalizing across a range of kinematic viscosity ($\nu$) values. This involved modifying the existing `main_precision.py` script to create `main_prmtrc.py`, where $\nu$ is treated as an additional input to the neural network rather than a discoverable parameter.

## Implementation Details

The implementation followed the guidelines outlined in `parametric-pinn-generalization-R1.md`. Key modifications in `main_prmtrc.py` included:

*   **Network Architecture**: The input layer of the neural network was expanded to accept four inputs: `(x, y, t, nu)`.
*   **Input Scaling**: The $\nu$ input was scaled to the range `[-1, 1]` using a min-max normalization based on `nu_min_train` and `nu_max_train`.
*   **Decoupling $\nu$**: The `log_nu_pinn` trainable variable was removed, and $\nu$ is now provided as an input tensor to the `compute_pde_residual` and `predict_velocity` functions.
*   **Data Generation**: During training, `nu` values for PDE collocation points are randomly sampled from the defined training range (`[0.01, 0.1]`).
*   **Training Process**: The two-stage Adam and L-BFGS-B optimization was adapted to handle the parametric input.

## Experiment Configuration

The initial experiment was run with the following parameters:

*   **Seed**: 1
*   **Adam Epochs**: 2000
*   **Data-Only Pre-training Epochs**: 10000
*   **Lambda Data Weight**: 1.0
*   **Lambda PDE Weight**: 1.0
*   **Number of PDE Points**: 60000
*   **Nu Training Range**: `[0.01, 0.1]`

## Results

The experiment completed successfully. The detailed log is available in `logs/parametric_run_seed_1.txt`.

### Performance Metrics

| Metric                       | Value        |
| :--------------------------- | :----------- |
| Data Preparation Duration    | 3.81 seconds |
| Model Initialization Duration| 4.34 seconds |
| Data-Only Pre-training Duration | 21.24 seconds |
| Adam Training Duration       | 324.35 seconds |
| L-BFGS-B Training Duration   | 461.15 seconds |
| **Total Execution Duration** | **821.29 seconds (13.69 minutes)** |

### Loss Evolution

*   **Data-Only Pre-training Loss**: Decreased from 0.093262 (Epoch 0) to 0.000031 (Epoch 9000).
*   **Adam Training (Full Loss)**: Decreased from 0.628549 (Epoch 0) to 0.001027 (Epoch 1900).

### Prediction Accuracy (for $\nu = 0.05$)

The model's predictions were evaluated against the ground truth data generated with $\nu = 0.05$.

*   **Prediction MSE (u)**: 3.592486e-02
*   **Prediction MSE (v)**: 3.803270e-02
*   **Total Prediction MSE (u+v)**: 7.395756e-02

## Discussion

This initial experiment successfully demonstrates the implementation and functionality of the parametric PINN. The model trained and converged, providing predictions for different $\nu$ values.

However, the total MSE of 0.0739 for $\nu = 0.05$ is significantly higher than the best results achieved in the precision-focused experiments (which reached errors as low as 0.044% relative error, implying much lower MSEs). This is an expected trade-off: a model that generalizes over a range of parameters typically has lower accuracy for any single parameter instance compared to a model specifically trained for that instance.

The current performance serves as a baseline. The next steps will focus on improving the generalization accuracy.

## Tuning Roadmap and HPC Considerations

To enhance the accuracy and efficiency of the parametric PINN, the following tuning roadmap is proposed:

1.  **Increase PDE Points**: Experiment with a higher number of PDE collocation points (e.g., 80,000, 100,000, 120,000). This directly increases the data available for the PDE loss, potentially improving the model's understanding of the underlying physics across the $\nu$ range.
    *   **HPC Impact**: This will increase memory usage and computational time for both Adam and L-BFGS-B phases. Monitor memory consumption and adjust batch sizes if necessary.

2.  **Increase Adam Epochs**: Extend the number of Adam training epochs (e.g., 3000, 5000). More epochs allow the model to further minimize the loss function.
    *   **HPC Impact**: Directly increases training time.

3.  **Neural Network Architecture**:
    *   **More Layers/Neurons**: Experiment with deeper or wider networks (e.g., `layers = [4, 80, 80, 80, 80, 80, 2]` or `layers = [4, 100, 100, 100, 100, 2]`). A larger network capacity might be needed to capture the more complex parametric solution space.
    *   **HPC Impact**: Increases memory usage and computational cost per epoch.

4.  **Adaptive Sampling**: Implement adaptive sampling strategies (as discussed in `parametric-pinn-generalization-R1.md`). This technique focuses computational resources on regions of high PDE residual, which is crucial for problems with sharp gradients or shocks (common in Burgers' equation, especially at lower $\nu$).
    *   **HPC Impact**: Can be computationally intensive due to the need for periodic error evaluation and dynamic point generation, but can lead to faster convergence to higher accuracy.

5.  **Explore $\nu$ Training Range**:
    *   **Wider Range**: Test if a wider range of $\nu$ values during training (e.g., `[0.005, 0.15]`) impacts generalization.
    *   **Denser Sampling**: Increase the density of $\nu$ values sampled during training, especially if certain $\nu$ regimes are more challenging.
    *   **HPC Impact**: Wider ranges might require more network capacity and longer training. Denser sampling increases the diversity of training data.

6.  **First-Order PINN (FO-PINN)**: If significant accuracy improvements are not achieved with the above, consider refactoring the problem to an FO-PINN formulation. This addresses numerical instabilities from high-order derivatives.
    *   **HPC Impact**: This is a more substantial change, potentially reducing computational cost per step by avoiding second-order automatic differentiation, but requires significant code modification.

Each step in this roadmap should be evaluated with new experiments, and the results documented in subsequent review files in `reviews/prmtrc/`. Performance metrics (time, MSE) should be consistently tracked to assess the impact of each tuning effort.

<br><sub>Last edited: 2025-08-18 16:20:00</sub>
