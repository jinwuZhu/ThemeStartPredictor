# test.py
import torch
from transformers import AutoTokenizer

from config import BERT_PRETRAINED_MODEL_NAME, HUGGINGFACE_CACHEDIR,TOKEN_MAX_LENGTH
from models import MiniThemeStartPredictor as ThemeStartPredictorModel
from utils import load_texts_by_folder

def main():
    max_length = TOKEN_MAX_LENGTH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./weights/theme_start_min_distilled.pt"
    test_text = load_texts_by_folder("./data/demo",endswidth=".md")[0]
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED_MODEL_NAME, cache_dir=HUGGINGFACE_CACHEDIR)
    # 初始化模型
    model = ThemeStartPredictorModel().to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    
    # 清理文本
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

if __name__ == "__main__":
    main()