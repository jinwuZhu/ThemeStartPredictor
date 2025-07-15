import torch
from transformers import AutoTokenizer
from config import BERT_PRETRAINED_MODEL_NAME, HUGGINGFACE_CACHEDIR, TOKEN_MAX_LENGTH
from models import ThemeStartPredictor

def load_model(model_path: str, device):
    """
    加载模型并设置为 eval 模式
    """
    model = ThemeStartPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_tokenizer():
    """
    加载 tokenizer
    """
    return AutoTokenizer.from_pretrained(BERT_PRETRAINED_MODEL_NAME, cache_dir=HUGGINGFACE_CACHEDIR)

def predict_theme_start(text: str, model, tokenizer, device, max_length=TOKEN_MAX_LENGTH) -> int:
    """
    预测主题开始位置（字符级索引）

    :param text: 原始纯文本（不含 <ai-theme> 标签）
    :param model: 已加载并设置为 eval() 的模型
    :param tokenizer: 对应的 tokenizer
    :param device: 运行设备（cuda / cpu）
    :param max_length: 最大 token 长度
    :return: 字符索引位置（char_start）
    """
    encoding = tokenizer(
        text,
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
    return char_start

# 🧪 示例运行方式（可选）
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="主题开始位置预测")
    parser.add_argument("--model_path", type=str, default="weights/theme_start_model.pt", help="模型路径")
    parser.add_argument("--text", type=str, help="待预测的文本内容（优先于文件）")
    parser.add_argument("--file", type=str, help="包含待预测文本的文件路径")

    args = parser.parse_args()

    if not args.text and not args.file:
        print("❌ 错误：必须提供 --text 或 --file 参数之一")
        exit(1)

    if args.text:
        text = args.text
    else:
        if not os.path.isfile(args.file):
            print(f"❌ 错误：文件不存在: {args.file}")
            exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    model = load_model(args.model_path, device)

    start_index = predict_theme_start(text, model, tokenizer, device)
    print("📍 预测字符起始位置:", start_index)
    print("📎 预测片段:", text[start_index:start_index + 125])
