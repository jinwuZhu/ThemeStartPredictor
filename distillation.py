# distillation.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import BERT_PRETRAINED_MODEL_NAME, TOKEN_MAX_LENGTH,HUGGINGFACE_CACHEDIR
from models import ThemeStartPredictor, MiniThemeStartPredictor
from dataset import ThemeDataset
from utils import load_texts_by_folder

# =====================
# 参数设置
# =====================
save_path = "weights/theme_min_hz128l8h4_1024.pt"
batch_size = 16
max_batches = 150
learning_rate = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED_MODEL_NAME,cache_dir=HUGGINGFACE_CACHEDIR)

# =====================
# 加载模型
# =====================
teacher = ThemeStartPredictor(pretrained_model_name=BERT_PRETRAINED_MODEL_NAME).to(device)
teacher.load_state_dict(torch.load("./weights/theme_start_model.pt"))
teacher.eval()

student = MiniThemeStartPredictor(hidden_size=128,num_layers=8,num_heads=4,max_len=1024).to(device)
student.train()

# =====================
# 加载数据集
# =====================
texts = load_texts_by_folder("./data/sources", endswidth=".md")
dataset = ThemeDataset(texts=texts, tokenizer=tokenizer, max_length=TOKEN_MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =====================
# 损失函数 & 优化器
# =====================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)

# =====================
# 蒸馏训练过程
# =====================
print("Start training...")
batch_count = 0

while batch_count < max_batches:
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 获取 teacher 的预测作为 soft label（不需要 ground-truth）
        with torch.no_grad():
            _, teacher_logits = teacher(input_ids, attention_mask)
            labels = teacher_logits.argmax(dim=-1)  # [batch_size]，每个句子的起始位置

        # student forward & loss
        optimizer.zero_grad()
        _, student_logits = student(input_ids, attention_mask)
        loss = loss_fn(student_logits, labels)
        loss.backward()
        optimizer.step()

        batch_count += 1
        print(f"\rBatch {batch_count:03d}/{max_batches} | Loss: {loss.item():.4f}", end="")

        if batch_count >= max_batches:
            break

print("\n训练完成 ✅")

# =====================
# 保存 student 模型
# =====================
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(student.state_dict(), save_path)
print(f"模型已保存至: {save_path}")
