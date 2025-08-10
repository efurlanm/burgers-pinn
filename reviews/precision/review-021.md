# Review 21: Experimental Design using Nested/Hierarchical Sampling

## 1. Objective

To ensure the robustness and generalizability of our findings for the academic paper, we are adopting a structured experimental design. This document outlines the methodology for this phase, which is based on Nested/Hierarchical Sampling.

## 2. Nested/Hierarchical Sampling Approach

We will structure our experiments in a two-level hierarchy. This approach ensures that we test each experimental condition independently and that the variations observed are due to the condition itself and not random chance.

-   **Level 1 (The "Nests"): Experimental Conditions.** We will define three distinct experimental conditions (nests) to test different aspects of the model’s performance: Robustness, Generalizability with a lower `nu`, and Generalizability with a higher `nu`.

-   **Level 2 (The "Samples"): Replications.** Within each experimental condition (nest), we will perform three independent replications. To ensure this independence, we will draw a unique set of three random seeds for each nest from a uniform distribution over the integers [1, 1000].

This can be visualized with the following diagram:

```mermaid
graph TD
    A[Experimental Design] --> B{Experiment 1: Robustness};
    A --> C{Experiment 2: Generalizability (Lower nu)};
    A --> D{Experiment 3: Generalizability (Higher nu)};

    B --> B1[Seed Sample 1];
    B --> B2[Seed Sample 2];
    B --> B3[Seed Sample 3];

    C --> C1[Seed Sample 1];
    C --> C2[Seed Sample 2];
    C --> C3[Seed Sample 3];

    D --> D1[Seed Sample 1];
    D --> D2[Seed Sample 2];
    D --> D3[Seed Sample 3];
```

## 3. Experimental Conditions

### Experiment 1: Robustness Analysis

-   **Objective:** To verify that the 0.044% relative error is a stable result and not an outlier.
-   **Methodology:** We will run the best-performing configuration (as verified in Review 20) with three different random seeds.
-   **Parameters:**
    -   `nu`: 0.05 (Ground Truth)
    -   `seeds`: We will use the existing result from `seed=1` as the first sample. We will then sample two additional unique seeds for the subsequent replications.

### Experiment 2: Generalizability with a Lower `nu`

-   **Objective:** To assess the model’s capability to discover a lower kinematic viscosity.
-   **Methodology:** We will modify the ground truth `nu` in the data generation process and run the PINN to discover it.
-   **Parameters:**
    -   `nu`: 0.01 (New Ground Truth)
    -   `seeds`: A new set of three unique seeds will be sampled.

### Experiment 3: Generalizability with a Higher `nu`

-   **Objective:** To assess the model’s capability to discover a higher kinematic viscosity.
-   **Methodology:** Similar to Experiment 2, but with a `nu` value an order of magnitude higher.
-   **Parameters:**
    -   `nu`: 0.1 (New Ground Truth)
    -   `seeds`: A new set of three unique seeds will be sampled.

## 4. Next Steps

We will now proceed with **Experiment 1: Robustness Analysis**. We will execute the simulation for two new seeds and append the results to this review file.

## 5. Experiment 1 Results: Robustness Analysis

We executed the experiment with three different seeds. The results are summarized below:

| Seed | Discovered `nu` | Relative Error |
| :--- | :-------------- | :------------- |
| 1    | 0.049978        | 0.044%         |
| 42   | 0.053882        | 7.764%         |
| 123  | 0.050855        | 1.71%          |

### 5.1. Discussion

The results clearly indicate that the model's performance is sensitive to the random seed. While the run with `seed=1` achieved a very high precision (0.044% error), the other two runs with `seed=42` and `seed=123` resulted in significantly higher errors (7.764% and 1.71%, respectively).

This finding is crucial. It suggests that the 0.044% error, while a valid result, may be an outlier or a consequence of a particularly favorable random initialization. For the academic paper, we cannot claim this level of precision as a general result. Instead, we must report the range of outcomes and discuss the model's sensitivity.

### 5.2. Next Steps

Given this sensitivity, it is not productive to proceed with Experiments 2 and 3 (Generalizability) at this moment. We first need to address the stability of the model. The next steps should focus on improving the model's robustness.

Potential strategies to improve robustness include:

-   **Ensemble Averaging:** Run the model with multiple seeds and average the discovered `nu` values. This is a common technique to reduce the variance of a model.
-   **Learning Rate Scheduling:** A more sophisticated learning rate schedule (e.g., with warm-up and decay) might help the model to find a more stable minimum.
-   **Network Architecture:** Experiment with different network architectures (e.g., more layers, different activation functions) to see if they lead to more stable results.

We will start by implementing ensemble averaging. We will run the experiment with a larger number of seeds (e.g., 10) and analyze the distribution of the results.

<br><sub>Last edited: 2025-08-10 12:08:08</sub>
