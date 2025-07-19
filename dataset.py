# dataset.py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils import extract_theme_span_labels

class ThemeDataset(Dataset):
    def __init__(self, texts, tokenizer: PreTrainedTokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text in texts:
            sample = extract_theme_span_labels(self.tokenizer, text,max_length=max_length)
            self.samples.append(sample)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "start_index": torch.tensor(item["start_index"], dtype=torch.long),
            "end_index": torch.tensor(item["end_index"], dtype=torch.long),
        }