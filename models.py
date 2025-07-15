# models.py
import torch
import torch.nn as nn
from transformers import BertModel
from config import HUGGINGFACE_CACHEDIR, BERT_PRETRAINED_MODEL_NAME


class ThemeStartLiner(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super(ThemeStartLiner, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()  # 也可用 ReLU
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)  # 输出 1 个分数

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x + residual)  # 加残差
        x = self.linear2(x)
        return x


class ThemeStartPredictor(nn.Module):
    def __init__(self, pretrained_model_name=BERT_PRETRAINED_MODEL_NAME, hidden_size=768):
        super(ThemeStartPredictor, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name,weights_only=False, cache_dir=HUGGINGFACE_CACHEDIR)
        self.linear = ThemeStartLiner(hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        logits = self.linear(last_hidden_state).squeeze(-1)  # (batch, seq_len)
        probs = self.softmax(logits)
        predicted_start_index = torch.argmax(probs, dim=1)  # (batch,)
        return predicted_start_index, logits  # 返回 logits 以便训练时用 CrossEntropyLoss


if __name__ == '__main__':
    model = ThemeStartPredictor()
    print(model)