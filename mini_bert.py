import torch
import torch.nn as nn
import torch.nn.functional as F

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len=512):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.segment_embeddings = nn.Embedding(2, hidden_size)  # segment A/B

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embed = self.token_embeddings(input_ids)
        pos_embed = self.position_embeddings(position_ids)
        seg_embed = self.segment_embeddings(token_type_ids)

        embeddings = word_embed + pos_embed + seg_embed
        return self.dropout(self.norm(embeddings))


class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, hidden_size),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class MiniBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=128, num_layers=2, num_heads=4, ff_size=512, max_len=512):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_len)
        self.encoder = nn.ModuleList([
            BertEncoderLayer(hidden_size, num_heads, ff_size) for _ in range(num_layers)
        ])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.encoder:
            x = layer(x, attention_mask)
        cls_output = x[:, 0]  # [CLS] 向量
        return x, cls_output
