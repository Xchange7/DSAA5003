"""
HPO Script: Optimize LoRA params for Mistral-7B using Ray Tune.
Optimized for 4x A800 (Parallel Execution).
No Web Dashboard - Logging to local text files.
"""

import os
import gc
import random
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import softmax # 或者使用 torch.nn.functional.softmax

from datasets import Dataset
from peft import (
get_peft_model,
LoraConfig,
TaskType,
)
from sklearn.metrics import accuracy_score
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
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

# 禁止 Tokenizer 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Config & Setup
# =========================

# ★★★ 修改点 1: 模型路径 ★★★
TARGET_MODEL = "/hpc2hdd/home/pxu364/dsaa/mistral-7b"

DATA_DIR = "data"

# ★★★ 修改点 2: 输出路径改名，避免和 DeepSeek 的结果混淆 ★★★
OUTPUT_ROOT = "output_hpo_mistral_2"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_FOLDS = 5
NUM_LABELS = 7
SEED = 42
N_TRIALS = 50 # 总试验次数

# 配置简单的文件日志
logging.basicConfig(
filename=os.path.join(OUTPUT_ROOT, "hpo_process.log"),
level=logging.INFO,
format='%(asctime)s - %(message)s',
datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 同时输出到控制台
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# =========================
# Utils
# =========================

def set_seed(seed: int = SEED) -> None:
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def preprocess_function(examples, tokenizer, max_length=512):
return tokenizer(
examples["input"],
truncation=True,
max_length=max_length,
padding=True,
)

# =========================
# 修改后的 compute_metrics
# =========================

def compute_metrics(eval_pred):
predictions, labels = eval_pred
# predictions 目前是 logits (原始分数)，需要转为概率
# axis=1 表示在每一个样本的 7 个类别上做 softmax
probs = softmax(predictions, axis=1)
# 计算 Log Loss
# labels 必须是整数 [0, 1, ... 6]，labels 参数告诉函数总共有几类
loss_val = log_loss(labels, probs, labels=list(range(NUM_LABELS)))
# 为了观察，我们也保留 accuracy，但主要优化目标是 loss
preds_hard = np.argmax(predictions, axis=1)
accuracy_val = accuracy_score(labels, preds_hard)
# 返回字典，键名 "log_loss" 将用于 Ray Tune 的监控
return {"log_loss": loss_val, "accuracy": accuracy_val}

# =========================
# Data Loading (Cached)
# =========================

def load_data():
logger.info(f"Loading Tokenizer from {TARGET_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False, trust_remote_code=True)
# Mistral 必须要设置 Pad Token，通常设为 EOS
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

logger.info("Loading Data...")
train_path = os.path.join(DATA_DIR, "train.csv")
df = pd.read_csv(train_path)
df["Response"] = df["Response"].fillna("NA")
df["input"] = "Question: " + df["Question"] + "; Answer: " + df["Response"]
df = df.rename(columns={"target": "label"})
folds = GroupKFold(n_splits=N_FOLDS)
df["fold"] = -1
for i, (_, test_index) in enumerate(folds.split(df, df["label"], groups=df["Question"])):
df.loc[test_index, "fold"] = i
# Fold 0
fold = 0
valid_df = df[df["fold"] == fold]
train_df = df[df["fold"] != fold]

train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

train_tokenized = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
valid_tokenized = valid_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
logger.info(f"Train size: {len(train_tokenized)}, Valid size: {len(valid_tokenized)}")
return train_tokenized, valid_tokenized, tokenizer

# =========================
# Ray Tune Callback
# =========================

class RayReportCallback(TrainerCallback):
def on_evaluate(self, args, state, control, metrics=None, **kwargs):
if metrics:
# 汇报 log_loss 给 Ray (注意：eval_ 前缀是 Trainer 自动加的)
tune.report({
"log_loss": metrics.get("eval_log_loss"),
"accuracy": metrics.get("eval_accuracy")
})

# =========================
# Training Function
# =========================

def train_func(config):
# 重设种子
set_seed(SEED + random.randint(0, 10000))

# 获取全局数据
train_ds = global_train_ds
valid_ds = global_valid_ds
tokenizer = global_tokenizer

# Model Config
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForSequenceClassification.from_pretrained(
TARGET_MODEL,
num_labels=NUM_LABELS,
quantization_config=bnb_config,
device_map="auto",
trust_remote_code=True,
attn_implementation="sdpa", # Mistral 兼容 SDPA
torch_dtype=torch.bfloat16
)
base_model.config.pretraining_tp = 1
base_model.config.pad_token_id = tokenizer.pad_token_id

# LoRA Config
# Mistral 的模块命名与 Llama 一致，这些 target_modules 是准确的
peft_config = LoraConfig(
r=config["lora_r"],
lora_alpha=config["lora_r"] * 2,
lora_dropout=config["lora_dropout"],
bias="none",
task_type=TaskType.SEQ_CLS,
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(base_model, peft_config)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# Training Args
args = TrainingArguments(
output_dir="local_output",
learning_rate=config["learning_rate"],
per_device_train_batch_size=16,
per_device_eval_batch_size=32,
gradient_accumulation_steps=config["grad_acc"],
max_grad_norm=1.0,
optim="paged_adamw_32bit",
lr_scheduler_type="cosine",
num_train_epochs=config["num_train_epochs"],
weight_decay=0.01,
eval_strategy="steps",
save_strategy="no",
eval_steps=50,
logging_steps=10,
report_to="none",
bf16=True,
fp16=False,
disable_tqdm=True,
)

trainer = Trainer(
model=model,
args=args,
train_dataset=train_ds,
eval_dataset=valid_ds,
tokenizer=tokenizer,
data_collator=data_collator,
compute_metrics=compute_metrics,
callbacks=[RayReportCallback()],
)

try:
trainer.train()
except Exception as e:
print(f"Error in trial: {e}")
raise e
del model, base_model, trainer
gc.collect()
torch.cuda.empty_cache()

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
# 初始化 Ray，禁用 Dashboard (适合后台运行)
ray.init(num_gpus=4, include_dashboard=False, ignore_reinit_error=True)
# 加载全局数据
global_train_ds, global_valid_ds, global_tokenizer = load_data()

# 搜索空间
search_space = {
"learning_rate": tune.loguniform(1e-5, 3e-4),
"lora_r": tune.choice([8, 16, 32, 64]),
"lora_dropout": tune.choice([0.05, 0.1]),
"grad_acc": tune.choice([1, 2]),
"num_train_epochs": tune.choice([2, 3]),
}

# 调度器 (ASHA) - 保持不变
scheduler = ASHAScheduler(
metric="accuracy",
mode="max",
max_t=10000,
grace_period=100,
reduction_factor=2
)

# ★★★ 修改点：移除了导致报错的 CLIReporter ★★★
# 因为你无法查看控制台，且 CLIReporter 在新版 API 中配置繁琐且易报错，
# 我们直接依靠下面的 CSV 和 Log 文件来记录结果，这样更稳健。

logger.info("Starting Ray Tune for Mistral-7B...")
logger.info(f"Results will be saved to: {os.path.abspath(OUTPUT_ROOT)}")

tuner = tune.Tuner(
tune.with_resources(
train_func,
resources={"gpu": 1, "cpu": 8}
),
param_space=search_space,
tune_config=tune.TuneConfig(
scheduler=scheduler,
num_samples=N_TRIALS,
),
# ★★★ 修改点：改为 tune.RunConfig，并添加 verbose=1 ★★★
run_config=tune.RunConfig(
storage_path=os.path.abspath(OUTPUT_ROOT),
name="mistral_asha_run",
verbose=1 # 强制设置为整数 1 (只输出基本信息)，防止 Ray 内部类型解析报错
)
)

# 开始训练
results = tuner.fit()

logger.info("Optimization Finished!")
# 获取最佳结果
best_result = results.get_best_result(metric="accuracy", mode="max")
logger.info(f"Best Accuracy: {best_result.metrics['accuracy']}")
logger.info(f"Best Config: {best_result.config}")
# 保存最佳参数
best_params_path = os.path.join(OUTPUT_ROOT, "best_params.json")
with open(best_params_path, "w") as f:
json.dump(best_result.config, f, indent=4)
logger.info(f"Saved best params to {best_params_path}")

# ★★★ 最重要的一步：保存所有试验结果到 CSV ★★★
# 这就是你离线查看所需的 Excel 表格
df = results.get_dataframe()
csv_path = os.path.join(OUTPUT_ROOT, "all_trials_results.csv")
df.to_csv(csv_path)
logger.info(f"Saved all trial results to {csv_path}")
