import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import BERT_PRETRAINED_MODEL_NAME, HUGGINGFACE_CACHEDIR, TOKEN_MAX_LENGTH
from dataset import ThemeStartDataset
from models import ThemeStartPredictor
from utils import load_texts_by_folder

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    max_length = TOKEN_MAX_LENGTH

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BERT_PRETRAINED_MODEL_NAME, cache_dir=HUGGINGFACE_CACHEDIR
    )

    # 加载训练数据
    train_seqs = load_texts_by_folder("./data/train", endswidth=".md")
    dataset = ThemeStartDataset(train_seqs, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 模型初始化
    model = ThemeStartPredictor().to(device)
    if args.resume and os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path))
        print(f"✅ 成功加载模型: {args.save_path}")
    else:
        print("🚀 初始化新模型")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    print("开始训练...\n")
    batch_count = 0
    while batch_count < args.max_batches:
        for batch in dataloader:
            if batch_count >= args.max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["start_index"].to(device)

            optimizer.zero_grad()
            _, logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            batch_count += 1
            print(f"\rBatch {batch_count:03d} | Loss: {loss.item():.4f}",end="")
    print("\n训练完成")
    # 保存模型
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\n✅ 模型已保存到: {args.save_path}")

    # 测试预测
    print("\n🔍 加载模型并进行测试预测：")
    model.load_state_dict(torch.load(args.save_path))
    model.eval()

    test_texts = load_texts_by_folder("./data/test", endswidth=".md")
    if not test_texts:
        print("⚠️ 没有找到测试文本")
        return

    test_text = test_texts[0]
    clean_text = test_text.replace("<ai-theme>", "")

    encoding = tokenizer(
        clean_text,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"][0]

    with torch.no_grad():
        pred_index, _ = model(input_ids, attention_mask)
        pred_index = pred_index.item()

    char_start = offsets[pred_index][0].item()
    print(f"\n🧪 原始文本: \n{test_text[:min(512,max_length)]}")
    print(f"📍 预测起始 token index: {pred_index}")
    print(f"📎 对应文本片段: \n{clean_text[char_start:char_start+125]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ThemeStartPredictor model.")
    parser.add_argument("--max_batches", type=int, default=100, help="训练批次数")
    parser.add_argument("--batch_size", type=int, default=2, help="训练批大小")
    parser.add_argument("--save_path", type=str, default="weights/theme_start_model.pt", help="模型保存路径")
    parser.add_argument("--resume", action="store_true", help="是否继续训练")

    args = parser.parse_args()
    main(args)
