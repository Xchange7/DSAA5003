# ⚡ Distributed HPO Strategy: Random Search with ASHA

To optimize the **DeepSeek-R1-Distill-Llama-8B** model efficiently, we first implement a hybrid strategy combining **Random Search** with the **Asynchronous Successive Halving Algorithm (ASHA)**.

See code RandomSearch

## 1. The Strategy: "Aggressive Early Stopping"

Standard Hyperparameter Optimization (HPO) suffers from a resource bottleneck: training Large Language Models (LLMs) is slow.
Instead of using complex Bayesian algorithms (which can be fragile in high-parallelism settings), we use **Random Search** augmented by **ASHA**.

### How it works in our pipeline:

1.  **Sampling (Random Search)**: 
    We sample **50 trials** from the hyperparameter space. Random search is chosen because it effectively covers high-dimensional spaces without the computational overhead of sequential Bayesian optimization.

2.  **Pruning (ASHA Scheduler)**: 
    This is the core efficiency driver. The scheduler monitors intermediate validation metrics (`Log Loss`) of all running trials.
    * **Promote:** Only the top-performing trials are allowed to continue training to the next epoch.
    * **Terminate:** Trials that underperform are **stopped early**, releasing their GPUs immediately to new random trials.

> **Impact:** This approach allows us to explore a wide range of hyperparameters while spending 80% of our GPU hours only on the most promising top 20% of configurations.

## 2. Search Space & Configuration

We utilize **Ray Tune** to define the following search space:

| Hyperparameter | Distribution | Logic |
| :--- | :--- | :--- |
| **Learning Rate** | `LogUniform(5e-5, 5e-4)` | Samples across orders of magnitude to find the optimal convergence speed. |
| **LoRA Rank (r)** | `Choice([8, 16, 32, 64])` | Tests different model capacities for the adapter layers. |
| **Grad Accumulation**| `Choice([1, 2])` | Tunes the effective batch size (Batch Size 16 * Grad Acc). |
| **Epochs** | `Choice([3, 4])` | Determines training duration. |

## 3. Technical Architecture

* **Framework:** Ray Tune + PyTorch
* **Parallelism:** 4 concurrent trials (1 GPU with 80GB VRAM per trial).
* **Memory Optimization:** Zero-copy data sharing via Ray Object Store (`ray.put`), ensuring high throughput.