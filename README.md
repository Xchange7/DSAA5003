# Hyperparameter Optimization (HPO) for LLM Fine-tuning

This repository contains a collection of scripts designed to optimize **LoRA (Low-Rank Adaptation)** hyperparameters for Large Language Models. The experiments focus on fine-tuning **DeepSeek-R1-Distill-Llama-8B** and **Mistral-7B** on a QA dataset.

We utilize two primary HPO frameworks—**Ray Tune**, **Optuna** **Hyperopt** and **OpenBox**—employing various search algorithms to benchmark performance and efficiency.

## ⚡ Experimental Setup & Hardware

These scripts are optimized for high-performance computing environments. Our reference experiments were conducted with the following specifications:

* **Hardware:** 4x NVIDIA A800 GPUs.
* **Concurrency:** 4 concurrent trials (1 trial per GPU).
* **Runtime:** Approximately **4 to 5 hours** per full experiment with 50 trials.
    * *Note: The runtime depends on the effectiveness of the early-stopping scheduler and the specific search space size.*

---

## 📂 File Overview & Differences

Each script represents a distinct combination of model, optimization framework, and search algorithm.

| File Name | Target Model | Framework | Search Algorithm | Scheduler | Key Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Hyperopt_ds.py`** | DeepSeek-R1 | Ray Tune | **HyperOpt (TPE)** | ASHA | Uses Tree-structured Parzen Estimator for intelligent, probability-based search. |
| **`RandomSearch_ds.py`** | DeepSeek-R1 | Ray Tune | **Random Search** | ASHA | Baseline approach using Ray's Object Store (`ray.put`) for data efficiency. |
| **`Openbox_ds.py`** | DeepSeek-R1 | **OpenBox** | **Bayesian Opt (GP)** | Async Parallel | Uses Gaussian Processes; manages GPU resources manually via `multiprocessing` queues. |
| **`RayTune_Mistral.py`** | Mistral-7B | Ray Tune | **Random Search** | ASHA | Specialized for Mistral; logs results to CSV files (Web Dashboard disabled). |

---

## 🧠 Algorithms & Methodologies Explained

### 1. Search Algorithms
The search algorithm determines **which hyperparameters to try next**.

* **Random Search** (Used in `RandomSearch_ds.py`, `RayTune_Mistral.py`):
    * **Mechanism:** Hyperparameters are sampled independently and uniformly from the defined search space.
    * **Pros:** Highly parallelizable and unbiased. Surprisingly effective for high-dimensional spaces.
    * **Cons:** Inefficient; it does not "learn" from previous results to find better configurations.

* **HyperOpt / TPE** (Used in `Hyperopt_ds.py`):
    * **Mechanism:** Uses the **Tree-structured Parzen Estimator (TPE)**. It models two probability distributions: one for "good" parameters and one for "bad" parameters. It then selects parameters that are likely to be "good" while exploring uncertain regions.
    * **Pros:** Converges faster than random search by focusing on promising areas.

* **OpenBox / Bayesian Optimization** (Used in `Openbox_ds.py`):
    * **Mechanism:** Uses **Gaussian Processes (GP)** as a surrogate model to approximate the loss function, combined with an **Acquisition Function** (e.g., Expected Improvement).
    * **Pros:** Theoretically the most sample-efficient method. OpenBox uses an **Asynchronous Parallel** strategy, allowing a new trial to start immediately on a free GPU without waiting for a batch to finish.

### 2. Scheduler: ASHA
Used in the Ray Tune scripts to speed up the process.
* **ASHA (Async Successive Halving Algorithm)** aggressively terminates (prunes) trials that are performing poorly early in the training process.
* **Benefit:** This allows the system to test many more configurations within the **4-5 hour** window compared to running every trial to completion.

---

## 🚀 How to Run

### 1. Prerequisites
Install the required Python libraries:

```bash
pip install torch transformers peft bitsandbytes datasets pandas scikit-learn scipy ray[tune] openbox
```


### 2. Data Preparation

The scripts expect a CSV file located at ../data/train.csv or data/train.csv. Required Columns:

**Question**: The input query.

**Response**: The desired answer.

**target**: The classification label (integer from 0 to 6).

### 3. Execution

A. DeepSeek with HyperOpt (TPE)

Recommended for finding the best result efficiently.

``` Bash
python src/Hyperopt_ds.py
```
B. DeepSeek with Random Search

A solid baseline comparison.

```Bash
python src/RandomSearch_ds.py
```

C. DeepSeek with OpenBox

Advanced Bayesian Optimization. Note: This script manages the GPU queue manually via multiprocessing. Ensure MAX_CONCURRENT_TRIALS matches your GPU count.

```Bash
python src/Openbox_ds.py
```

D. Mistral with Random Search

Optimized for Mistral-7B architecture.

```Bash
python src/RayTune_Mistral.py
```

## ⚙️ Configuration Notes
If adapting these scripts for different hardware or models, pay attention to these variables:

 - **TARGET_MODEL**: Update the absolute path to your local model weights.

 -  **MAX_CONCURRENT_TRIALS** (OpenBox) or resources={"gpu": 1} (Ray):

Set this based on your available GPUs.

For our setup (4x A800), this is set to allow 4 parallel jobs.

 - **search_space**: The dictionary defining the range of hyperparameters to explore (e.g., lora_r values of [8, 16, 32, 64]).

 - **OUTPUT_ROOT**: Defines where logs and checkpoints are saved.

## 📊 Results
**Logs**: Training logs (Loss/Accuracy) are saved to the defined OUTPUT_ROOT directory.

**Results**: The Mistral script explicitly exports all_trials_results.csv for offline analysis.

## 🚀 Final Training & Inference Pipeline

After identifying the optimal hyperparameters via the HPO stage (Ray Tune/OpenBox), we execute a two-stage pipeline to generate the final model and submission predictions.

### 1. Retraining with Best Hyperparameters

We retrain the DeepSeek-R1-Distill-Llama-8B model using the global best configuration found during our experiments.**Script:** `train_final.py`

This script:

- Loads the pre-trained 4-bit Quantized model.
- Applies the optimal LoRA configuration (Rank=32, Alpha=64, Dropout=0.1).
- Uses the specific learning rate (2.35×10−4) and Cosine scheduler found by our best Random Search trial.
- Saves the final adapter weights to `./final_best_model_output/best_model`.

**Usage:**

Before running, please ensure you update the `TARGET_MODEL` and `DATA_DIR` paths in the script to match your local environment.

```bash
python train_final.py
```

### 2. Inference & Submission

Once the model is retrained, we run inference on the test dataset to generate the `submission.csv` file containing probability distributions for the 7 target classes. **Script:** `inference.py`

This script:

- Merges the base model with the trained LoRA adapter from Step 1.
- Tokenizes the test data (`test.csv`).
- Performs batched inference (Batch Size=32) for efficiency.
- Outputs the results to `submission.csv`.

**Usage:**

```bash
python inference.py
```

[//]: # (**JSON**: OpenBox exports optimization history to openbox_results.json.)