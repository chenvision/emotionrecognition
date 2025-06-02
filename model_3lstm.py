"""BiLSTM classifier with self-attention mechanism + LayerNorm."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["BiLSTMClassifier3"]

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, H]
        attention_weights = F.softmax(self.attention(x), dim=1)  # [B, T, 1]
        attended = torch.bmm(x.transpose(1, 2), attention_weights)  # [B, H, 1]
        return attended.squeeze(-1), attention_weights  # [B, H], [B, T, 1]

class BiLSTMClassifier3(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_class: int = 2, lstm_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 嵌入层
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,  # 3层LSTM
            bidirectional=True,
            batch_first=True,
            dropout=dropout  # 只有 num_layers > 1 时生效
        )
        
        # 加一个 LayerNorm，归一化LSTM输出
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # 双向LSTM，2倍hidden_dim
        
        self.attention = SelfAttention(hidden_dim * 2)  # 注意力机制
        self.dropout = nn.Dropout(dropout)  # 外层Dropout
        self.fc = nn.Linear(hidden_dim * 2, num_class)  # 全连接分类层

    def forward(self, x):
        emb = self.embedding(x)  # [B, T, E]
        lstm_out, _ = self.lstm(emb)  # [B, T, 2H]
        lstm_out = self.layer_norm(lstm_out)  # LayerNorm [B, T, 2H]
        
        attended, attention_weights = self.attention(lstm_out)  # [B, 2H], [B, T, 1]
        out = self.dropout(attended)  # Dropout
        logits = self.fc(out)  # [B, num_class]
        return logits, attention_weights
