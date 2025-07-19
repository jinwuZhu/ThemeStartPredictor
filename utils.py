# uitls.py
import os
def extract_theme_span_labels(tokenizer, text: str, max_length: int = 8192,start_tag="<ai-theme>",end_tag="</ai-theme>"):
    theme_start = start_tag
    theme_end = end_tag
    
    start_index = 0
    end_index = 0

    start_pos = text.find(theme_start)
    end_pos = text.find(theme_end)
    # 清洗标签
    clean_text = text.replace(theme_start, "").replace(theme_end, "")

    encoding = tokenizer(
        clean_text,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    if start_pos != -1 and end_pos != -1:
        if start_pos > end_pos:
            print(f"[ERROR] Invalid tag order: <ai-theme> after </ai-theme> in: '{text[:60]}...'")
            return None

        # 计算 span 范围（去除标签对之后的实际内容起止字符位置）
        span_char_start = start_pos
        span_char_end = end_pos - len(theme_start)

        offsets = encoding["offset_mapping"][0]  # shape: (seq_len, 2)

        # 查找 start_index
        for i, (start_char, _) in enumerate(offsets):
            if start_char >= span_char_start:
                start_index = i
                break

        # 查找 end_index
        for i in range(len(offsets) - 1, -1, -1):
            end_char = offsets[i][1]
            if end_char <= span_char_end:
                end_index = i
                break

    return {
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "start_index": start_index,
        "end_index": end_index
    }


def load_texts_by_folder(folder_path: str,endswidth=".txt") -> list[str]:
    filelist = os.listdir(folder_path)
    resuls = []
    for file in filelist:
        if file.endswith(endswidth):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                resuls.append(f.read())
    return resuls
    