# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import BERT_PRETRAINED_MODEL_NAME, HUGGINGFACE_CACHEDIR,TOKEN_MAX_LENGTH
from dataset import ThemeStartDataset
from models import ThemeStartPredictor
from utils import load_texts_by_folder

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
max_length = TOKEN_MAX_LENGTH
# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED_MODEL_NAME, cache_dir=HUGGINGFACE_CACHEDIR)

# 构造训练样本
train_seqs = [
]
train_seqs.extend(load_texts_by_folder("./data/train",endswidth=".md"))
# 构造 Dataset 和 DataLoader
dataset = ThemeStartDataset(train_seqs, tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 模型、loss、优化器
model = ThemeStartPredictor().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 比 Adam 更稳定（带权重衰减）

# 训练模式
model.train()

max_batches = 100
batch_count = 0

while batch_count < max_batches:
    for batch in dataloader:
        if batch_count >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["start_index"].to(device)

        optimizer.zero_grad()
        _, logits = model(input_ids, attention_mask)  # logits shape: (batch, seq_len)
        loss = loss_fn(logits, labels)  # labels shape: (batch,)

        loss.backward()
        optimizer.step()

        print(f"Batch {batch_count + 1:03d} | Loss: {loss.item():.4f}")
        batch_count += 1

model_path = "weights/theme_start_model.pt"
torch.save(model.state_dict(), model_path)
print(f"\n✅ 模型已保存到: {model_path}")

# ✅ 加载模型 & 推理测试
print("\n🔍 加载模型并进行测试预测：")
model.load_state_dict(torch.load(model_path))
model.eval()

# 测试句子
test_text = load_texts_by_folder("./data/test",endswidth=".md")[0]

# 去除 <theme_start> 后 tokenize
clean_text = test_text.replace("<ai-theme>", "")

encoding = tokenizer(
    clean_text,
    return_offsets_mapping=True,
    padding='max_length',
    truncation=True,
    max_length=max_length,
    return_tensors='pt'
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)
offsets = encoding["offset_mapping"][0]

with torch.no_grad():
    pred_index, _ = model(input_ids, attention_mask)
    pred_index = pred_index.item()

# 打印预测结果
char_start = offsets[pred_index][0].item()
print(f"\n🧪 原始文本: \n{test_text[:256]}")
print(f"📍 模型预测主要内容开始位置 token_index: {pred_index}")
print(f"📎 对应文本内容: \n{clean_text[char_start:char_start+125]}")