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

# import wandb
# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_PROJECT"] = "Deepseek--RandomSearch"

os.environ["WANDB_DISABLED"] = "true"

# 防止 Tokenizer 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Config
# =========================
TARGET_MODEL = "/data/user/pxu364/models/DeepSeek-R1-Distill-Llama-8B"
DATA_DIR = "data"
OUTPUT_ROOT = "output_hpo_deepseek_advanced"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_FOLDS = 5
NUM_LABELS = 7
SEED = 42
N_TRIALS = 50

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
    # 预防 tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # 预防 float16 溢出
    predictions = predictions.astype(np.float32)

    # Softmax
    probs = softmax(predictions, axis=1)

    # 截断防止 inf
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    # 假设 NUM_LABELS = 7
    label_list = list(range(NUM_LABELS))

    try:
        # labels 参数告诉 sklearn 总共有哪些类，防止 batch 里缺类导致报错
        loss_val = log_loss(labels, probs, labels=label_list)
    except ValueError as e:
        print(f"Log Loss Error: {e}")
        # 如果报错，说明 label 值超出了 0-6 的范围
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
    tokenizer.add_eos_token = True

    logger.info("Loading Data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["Response"] = df["Response"].fillna("NA")

    # Prompt 优化
    # 使用伪 Chat 格式，让模型更清晰地理解这是 User 提问和 Assistant 回答
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

def train_func(config):
    set_seed(SEED + random.randint(0, 10000))

    train_df = ray.get(config["train_df_ref"])
    valid_df = ray.get(config["valid_df_ref"])
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 动态分词
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

    # 4-bit 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        TARGET_MODEL,
        num_labels=NUM_LABELS,
        quantization_config=bnb_config,
        device_map="auto",  # Ray 会自动分配 CUDA_VISIBLE_DEVICES
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


    model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir="local_output",
        learning_rate=config["learning_rate"],

        # ★ Batch Size 改小了 (16)
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,  # 验证集也改小，防止验证时OOM

        # ★ 这里的 grad_acc 会乘以 ray tune 搜索到的 config["grad_acc"]
        # 如果 config["grad_acc"] 搜出来是 1，实际 accumulation 就是 1
        # 建议为了模拟大 Batch，这里可以在 config 外面再乘一个系数，或者直接依赖 config
        gradient_accumulation_steps=config["grad_acc"] * 4,  # ★ 强行累加4次，凑回 Batch 64 的效果

        # ★★★ 开启 Gradient Checkpointing (显存救星) ★★★
        gradient_checkpointing=True,

        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        label_smoothing_factor=config["label_smoothing_factor"],
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",

        # save_strategy="no",
        eval_strategy="steps",
        eval_steps=20,

        save_strategy="steps",
        save_steps=20,  # 步数建议与 eval 一致
        save_total_limit=1,  # 只保留 1 个最新的 checkpoint

        load_best_model_at_end=True,  # ★ 训练结束时自动回滚到表现最好的模型
        metric_for_best_model="log_loss",
        greater_is_better=False,  # loss 越小越好

        logging_steps=10,
        report_to="none",  # 可选wandb
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

    # 清理
    del model, base_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 初始化 Ray
    ray.init(num_gpus=4, include_dashboard=False)

    # 1. 加载数据
    t_df, v_df, _ = load_raw_data_and_tokenizer()

    # 2. 将数据放入 Ray 对象存储 (Object Store)
    train_df_ref = ray.put(t_df)
    valid_df_ref = ray.put(v_df)

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

        "train_df_ref": train_df_ref,
        "valid_df_ref": valid_df_ref
    }

    scheduler = ASHAScheduler(
        metric="log_loss",
        mode="min",
        max_t=1000,
        grace_period=20,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_func,
            resources={"gpu": 1, "cpu": 4}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=N_TRIALS,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath(OUTPUT_ROOT),
            name="deepseek_h100_run",
        )
    )

    results = tuner.fit()
    print("Best config: ", results.get_best_result(metric="log_loss", mode="min").config)