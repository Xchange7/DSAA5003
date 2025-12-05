import os
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# =========================
# 配置
# =========================
BASE_MODEL_PATH = "/data/user/pxu364/models/DeepSeek-R1-Distill-Llama-8B"
# 确保这个路径是你 train_final.py 跑完生成的那个文件夹
ADAPTER_PATH = "final_best_model_output/best_model"
TEST_DATA = "../data/test.csv"
OUTPUT_FILE = "submission.csv"

# 推理时显存压力小，Batch Size 可以大一点
BATCH_SIZE = 32


def format_text(row):
    # 必须和训练时完全一致
    return f"<|User|>: {row['Question']}\n<|Assistant|>: {row['Response']}"


def main():
    print("1. Loading Tokenizer and Base Model...")
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    # 【修复重点 1】确保 Tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # 推理必须用 right padding

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=7,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"2. Loading LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("3. Preparing Test Data...")
    test_df = pd.read_csv(TEST_DATA)

    # 处理缺失值（如果有）
    test_df["Response"] = test_df["Response"].fillna("NA")

    # 应用格式化
    test_df["input"] = test_df.apply(format_text, axis=1)
    texts = test_df["input"].tolist()

    print(f"4. Starting Inference on {len(texts)} samples...")
    all_probs = []

    # 手动 Batch 推理
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.float().cpu().numpy()
            probs = softmax(logits, axis=1)
            all_probs.append(probs)

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   Processed {min(i + BATCH_SIZE, len(texts))} / {len(texts)}")

    # 拼接结果
    final_probs = np.concatenate(all_probs, axis=0)

    # 生成提交文件
    cols = [f"target_{i}" for i in range(7)]
    submission = pd.DataFrame(final_probs, columns=cols)

    # 如果 test.csv 有 id 列，加上 id
    if "id" in test_df.columns:
        submission.insert(0, "id", test_df["id"])

    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done! Submission saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()