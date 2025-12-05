import os
import gc
import random
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import multiprocessing
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

# OpenBox Imports
from openbox import Optimizer, sp

# =========================
# Config
# =========================
TARGET_MODEL = "/hpc2hdd/home/pxu364/models/DeepSeek-R1-Distill-Llama-8B"
DATA_DIR = "../data"
OUTPUT_ROOT = "output_hpo_deepseek_openbox"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_FOLDS = 5
NUM_LABELS = 7
SEED = 42
N_TRIALS = 50
MAX_CONCURRENT_TRIALS = 4  # ★你有4张卡，这里设置为4，实现并行

# H100 优化
BATCH_SIZE = 16

# 防止 Tokenizer 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

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
    tokenizer.add_eos_token = True

    logger.info("Loading Data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
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


# =========================
# Global Resource Manager
# =========================
# 用于并行训练时分配 GPU ID
gpu_queue = multiprocessing.Manager().Queue()
for i in range(MAX_CONCURRENT_TRIALS):
    gpu_queue.put(i)

# 全局数据，利用 Linux Fork 机制让子进程可读，避免传递
GLOBAL_TRAIN_DF = None
GLOBAL_VALID_DF = None


# =========================
# Objective Function
# =========================

def objective_function(config):
    """
    OpenBox 调用的目标函数
    Config 是一个字典，包含当次采样的超参数
    """
    # 1. 获取 GPU 资源
    try:
        gpu_id = gpu_queue.get(timeout=3600)  # 等待资源
    except Exception as e:
        logger.error("Failed to get GPU resource")
        return {'objectives': [999.0]}  # 返回极大 Loss

    # 设置当前进程只能看到这块 GPU
    # 注意：这必须在 torch 初始化 CUDA context 之前生效
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 打印当前进程信息
    pid = os.getpid()
    print(f"Process {pid} running on GPU {gpu_id} with config: {config}")

    loss_result = 999.0  # 默认失败值

    try:
        set_seed(SEED + random.randint(0, 10000))

        # 直接使用全局变量（Read-only）
        train_df = GLOBAL_TRAIN_DF
        valid_df = GLOBAL_VALID_DF

        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        MAX_LEN = 1024

        def _tokenize(examples):
            return tokenizer(examples["input"], truncation=True, max_length=MAX_LEN, padding=False)

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
            # ★ 关键：因为前面设置了 CUDA_VISIBLE_DEVICES，这里 auto 会映射到唯一的可见卡
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

        model.enable_input_require_grads()

        args = TrainingArguments(
            output_dir=f"{OUTPUT_ROOT}/trial_{pid}",
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
            metric_for_best_model="log_loss",
            greater_is_better=False,
            logging_steps=10,
            report_to="none",
            bf16=True,
            fp16=False,
            disable_tqdm=True,
            dataloader_num_workers=0,  # 并行进程中设为0更安全
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
        )

        trainer.train()

        # 获取最佳 Loss
        # Trainer 会自动加载最佳模型，所以 evaluate 结果就是最佳结果
        metrics = trainer.evaluate()
        loss_result = metrics["eval_log_loss"]

        # 清理显存
        del model, base_model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in trial: {e}")
        loss_result = 999.0
    finally:
        # ★ 关键：释放 GPU 资源回到队列
        gpu_queue.put(gpu_id)
        # 强制垃圾回收
        gc.collect()

    # OpenBox 期望返回一个字典，objectives 是目标值列表（支持多目标，这里是单目标）
    return {'objectives': [loss_result]}


# =========================
# Main Driver
# =========================

if __name__ == "__main__":
    # 1. 准备数据 (加载到全局变量)
    t_df, v_df, _ = load_raw_data_and_tokenizer()
    GLOBAL_TRAIN_DF = t_df
    GLOBAL_VALID_DF = v_df

    print("Data loaded. Defining search space...")

    # 2. 定义搜索空间 (OpenBox Style)
    # OpenBox 的 API 与 Ray Tune 略有不同
    space = sp.Space()
    space.add_variables([
        sp.Real("learning_rate", 5e-5, 5e-4, log=True),
        sp.Categorical("num_train_epochs", [3, 4]),
        sp.Categorical("grad_acc", [1, 2]),
        sp.Categorical("lora_r", [8, 16, 32, 64]),
        sp.Categorical("lora_dropout", [0.05, 0.1]),
        sp.Real("label_smoothing_factor", 0.0, 0.1),
        sp.Categorical("weight_decay", [0.01, 0.1]),
        sp.Real("warmup_ratio", 0.0, 0.1),
        sp.Categorical("lr_scheduler_type", ["cosine", "linear"]),
    ])

    print("Starting OpenBox Optimizer...")

    # 3. 定义优化器
    # 使用 ParallelOptimizer 进行并行搜索
    opt = Optimizer(
        objective_function,
        space,
        num_objectives=1,
        num_constraints=0,
        max_runs=N_TRIALS,
        surrogate_type='gp',  # 默认高斯过程，也可以选 'prf' (随机森林)
        acq_type='ei',  # 采集函数
        acq_optimizer_type='local_random',
        initial_runs=5,  # 初始随机探索次数
        task_id='deepseek_hpo',
        # 并行设置
        # 注意：OpenBox 的并行需要 backend 支持，
        # 简单的多进程可以使用 batch_size + async 策略，
        # 但标准的 Optimizer 类通常是串行的。
        # 这里我们使用 batch_size 来触发 OpenBox 内部的并行建议生成，
        # 并结合 multiprocessing (上面的 objective_function 逻辑) 来实现真正的并行。
        # *实际上 OpenBox 有 ParallelOptimizer 类，用法如下*
    )

    # 重新定义为 ParallelOptimizer 以更好支持多卡
    from openbox import ParallelOptimizer

    opt = ParallelOptimizer(
        objective_function,
        space,
        parallel_strategy='async',  # 异步并行，一有空闲 GPU 就跑
        batch_size=MAX_CONCURRENT_TRIALS,
        batch_strategy='default',
        num_objectives=1,
        num_constraints=0,
        max_runs=N_TRIALS,
        surrogate_type='gp',
        task_id='deepseek_hpo_parallel',
    )

    # 4. 运行
    history = opt.run()

    # 5. 结果
    print("Optimization finished.")
    print("History:", history)

    # 获取最佳配置
    best_config = history.get_incumbents()[0]
    print("Best config:", best_config)
    print("Best objective value:", history.get_incumbents()[0].perf)

    # 保存结果
    with open(os.path.join(OUTPUT_ROOT, "openbox_results.json"), "w") as f:
        json.dump(str(history), f)