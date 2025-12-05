import os
import gc
import random
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

# 防止 Tokenizer 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Config
# =========================
TARGET_MODEL = "/data/user/pxu364/models/DeepSeek-R1-Distill-Llama-8B"
DATA_DIR = "../data"
OUTPUT_ROOT = "output_deepseek_hyperopt"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_FOLDS = 5
NUM_LABELS = 7
SEED = 42
N_TRIALS = 20
BATCH_SIZE = 16

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================
# Utils & Data
# =========================

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = predictions.astype(np.float32)
    probs = softmax(predictions, axis=1)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    label_list = list(range(NUM_LABELS))

    try:
        loss_val = log_loss(labels, probs, labels=label_list)
    except ValueError as e:
        print(f"Log Loss Error: {e}")
        loss_val = 99.0

    preds_hard = np.argmax(predictions, axis=1)
    accuracy_val = accuracy_score(labels, preds_hard)

    return {"log_loss": loss_val, "accuracy": accuracy_val}


def load_raw_data_and_tokenizer():
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    logger.info("Loading Data...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    except FileNotFoundError:
        logger.error(f"FATAL: Train data not found at {os.path.join(DATA_DIR, 'train.csv')}")
        raise

    df["Response"] = df["Response"].fillna("NA")

    def format_text(row):
        return f"<|User|>: {row['Question']}\n<|Assistant|>: {row['Response']}"

    df["input"] = df.apply(format_text, axis=1)
    df = df.rename(columns={"target": "label"})

    folds = GroupKFold(n_splits=N_FOLDS)
    df["fold"] = -1
    for i, (_, test_index) in enumerate(folds.split(df, df["label"], groups=df["Question"])):
        df.loc[test_index, "fold"] = i

    valid_df = df[df["fold"] == 0].reset_index(drop=True)
    train_df = df[df["fold"] != 0].reset_index(drop=True)

    return train_df, valid_df, tokenizer


class RayReportCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            tune.report({
                "log_loss": metrics.get("eval_log_loss"),
                "accuracy": metrics.get("eval_accuracy")
            })


# =========================
# Train Func
# =========================

def train_func(config, train_df=None, valid_df=None):
    set_seed(SEED + random.randint(0, 10000))

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    MAX_LEN = 1024

    def _tokenize(examples):
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )

    train_ds = Dataset.from_pandas(train_df).map(_tokenize, batched=True)
    valid_ds = Dataset.from_pandas(valid_df).map(_tokenize, batched=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        TARGET_MODEL,
        num_labels=NUM_LABELS,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_r"] * 2,
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, peft_config)

    base_model.score.weight.data.normal_(mean=0.0, std=0.01)
    if base_model.score.bias is not None:
        base_model.score.bias.data.zero_()

    os.environ["WANDB_DISABLED"] = "true"

    model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir="local_output",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=config["grad_acc"] * 4,
        gradient_checkpointing=True,
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        label_smoothing_factor=config["label_smoothing_factor"],
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",

        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_log_loss",
        greater_is_better=False,

        logging_steps=10,
        report_to="none",
        bf16=True,
        fp16=False,
        disable_tqdm=True,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        compute_metrics=compute_metrics,
        callbacks=[RayReportCallback()],
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Error: {e}")
        raise e

    del model, base_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ray.init(num_gpus=4, include_dashboard=False)

    # 1. 加载数据
    t_df, v_df, _ = load_raw_data_and_tokenizer()

    # 2. 定义搜索空间
    search_space = {
        "learning_rate": tune.loguniform(5e-5, 5e-4),
        "num_train_epochs": tune.choice([3, 4]),
        "grad_acc": tune.choice([1, 2]),
        "lora_r": tune.choice([8, 16, 32, 64]),
        "lora_dropout": tune.choice([0.05, 0.1]),
        "label_smoothing_factor": tune.uniform(0.0, 0.1),
        "weight_decay": tune.choice([0.01, 0.1]),
        "warmup_ratio": tune.uniform(0.0, 0.1),
        "lr_scheduler_type": tune.choice(["cosine", "linear"]),
    }

    hyperopt_search = HyperOptSearch(
        metric="log_loss",
        mode="min",
    )
    algo = ConcurrencyLimiter(hyperopt_search, max_concurrent=4)

    scheduler = ASHAScheduler(
        metric="log_loss",
        mode="min",
        max_t=1000,
        grace_period=20,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func, train_df=t_df, valid_df=v_df),
            resources={"gpu": 1, "cpu": 4}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=algo,
            scheduler=scheduler,
            num_samples=N_TRIALS,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(OUTPUT_ROOT),
            name="deepseek_hyperopt_run",
        )
    )

    results = tuner.fit()
    print("Best config: ", results.get_best_result(metric="log_loss", mode="min").config)