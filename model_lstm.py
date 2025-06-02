import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["BiLSTMClassifier"]


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, H]
        attention_weights = F.softmax(self.attention(x), dim=1)  # [B, T, 1]
        attended = torch.bmm(x.transpose(1, 2), attention_weights)  # [B, H, 1]
        return attended.squeeze(-1), attention_weights  # [B, H], [B, T, 1]


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_class: int = 2,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # 仅在 num_layers > 1 时生效
        )

        if use_attention:
            self.attention = SelfAttention(hidden_dim * 2)

        self.fc = nn.Linear(hidden_dim*2, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T]
        emb = self.embedding(x)  # [B, T, E]
        emb = self.norm(emb)
        out, _ = self.lstm(emb)  # [B, T, 2H]
        # out = out + emb

        if self.use_attention:
            pooled, attention_weights = self.attention(out)  # [B, 2H], [B, T, 1]
        else:
            pooled = torch.mean(out, dim=1)  # [B, 2H]
            attention_weights = None

        out = self.dropout(pooled)
        logits = self.fc(out)

        return logits, attention_weights
