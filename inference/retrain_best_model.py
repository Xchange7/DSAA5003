import os
import gc
import random
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# =========================
# 1. 核心配置 & 最佳参数
# =========================
# 模型路径
TARGET_MODEL = "/data/user/pxu364/models/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR = "final_best_model_output"
DATA_DIR = "../data"

# ★★★ 你的最佳参数 Best Params ★★★
BEST_PARAMS = {
    "learning_rate": 0.00023517471650108732,
    "num_train_epochs": 3,
    "grad_acc": 1,
    "lora_r": 32,
    "lora_dropout": 0.1,
    "label_smoothing_factor": 0.07103036556537999,
    "weight_decay": 0.01,
    "warmup_ratio": 0.08181384977671685,
    "lr_scheduler_type": "cosine"
}

# 环境设置
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BATCH_SIZE = 16
NUM_LABELS = 7
SEED = 42


# =========================
# 2. 工具函数
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
    except ValueError:
        loss_val = 99.0
    preds_hard = np.argmax(predictions, axis=1)
    accuracy_val = accuracy_score(labels, preds_hard)
    return {"log_loss": loss_val, "accuracy": accuracy_val}


def format_text(row):
    # 保持与 HPO 时完全一致的 Prompt 格式
    return f"<|User|>: {row['Question']}\n<|Assistant|>: {row['Response']}"


# =========================
# 3. 主训练流程
# =========================
def main():
    set_seed(SEED)
    print(f"Starting training with params: {BEST_PARAMS}")

    # --- 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- 加载数据 ---
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["Response"] = df["Response"].fillna("NA")
    df["input"] = df.apply(format_text, axis=1)
    df = df.rename(columns={"target": "label"})

    folds = GroupKFold(n_splits=5)
    df["fold"] = -1
    for i, (_, test_index) in enumerate(folds.split(df, df["label"], groups=df["Question"])):
        df.loc[test_index, "fold"] = i

    valid_df = df[df["fold"] == 0].reset_index(drop=True)
    train_df = df[df["fold"] != 0].reset_index(drop=True)

    def _tokenize(examples):
        return tokenizer(examples["input"], truncation=True, max_length=1024, padding=False)

    print("Tokenizing datasets...")
    train_ds = Dataset.from_pandas(train_df).map(_tokenize, batched=True)
    valid_ds = Dataset.from_pandas(valid_df).map(_tokenize, batched=True)

    # --- 模型准备 ---
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
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=BEST_PARAMS["lora_r"],
        lora_alpha=BEST_PARAMS["lora_r"] * 2,
        lora_dropout=BEST_PARAMS["lora_dropout"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, peft_config)

    # 初始化 Head
    base_model.score.weight.data.normal_(mean=0.0, std=0.01)
    if base_model.score.bias is not None:
        base_model.score.bias.data.zero_()
    model.enable_input_require_grads()

    # --- Trainer 配置 ---
    # 注意：这里还原了你之前的逻辑：grad_acc * 4
    real_grad_acc = BEST_PARAMS["grad_acc"] * 4

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=BEST_PARAMS["learning_rate"],
        num_train_epochs=BEST_PARAMS["num_train_epochs"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=real_grad_acc,
        weight_decay=BEST_PARAMS["weight_decay"],
        warmup_ratio=BEST_PARAMS["warmup_ratio"],
        lr_scheduler_type=BEST_PARAMS["lr_scheduler_type"],
        label_smoothing_factor=BEST_PARAMS["label_smoothing_factor"],

        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_log_loss",
        greater_is_better=False,

        logging_steps=10,
        report_to="none",
        bf16=True,
        dataloader_num_workers=4,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
        compute_metrics=compute_metrics,
    )

    # --- 开始训练 ---
    print("Starting Training...")
    trainer.train()

    # --- 保存最终模型 ---
    print(f"Training finished. Saving best model to {OUTPUT_DIR}/best_model ...")

    # 创建专门的子目录保存最终可用的 Adapter
    final_save_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    print("✅ Model saved successfully!")

    # 清理内存
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()