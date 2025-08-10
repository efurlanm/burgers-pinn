## Review 025

### Analysis of the Results

I attempted to run the `main_scipy.py` script with the number of hidden layers increased to 6. However, the script failed with a `ResourceExhaustedError`, which means the GPU ran out of memory.

### Proposed Next Steps

Since I cannot increase the model's complexity or the number of PDE points due to memory constraints, I will try to improve the training process itself. I will increase the number of Adam epochs from 2000 to 5000. This will give the Adam optimizer more time to find a good starting point for the L-BFGS-B optimizer, which may help it to find a better minimum.

<br><sub>Last edited: 2025-08-06 12:19:44</sub>
