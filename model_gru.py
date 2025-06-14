"""BiLSTM classifier with self-attention mechanism."""
# BiLSTM+Self-Attention的文本分类器
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["BiGRUClassifier"]


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, H]
        attention_weights = F.softmax(self.attention(x), dim=1)  # [B, T, 1]
        attended = torch.bmm(x.transpose(1, 2), attention_weights)  # [B, H, 1]
        return attended.squeeze(-1), attention_weights  # [B, H], [B, T, 1]
        # 对每个token分配一个权重，再加权求和，得到一个固定维度的句子表示。
        # 用于捕捉文本中的关键信息，提高模型的表达能力和性能。


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_class: int = 2, dropout: float = 0.1):
        super().__init__()  # 词表大小，嵌入维度，隐藏维度，类别数
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 嵌入层
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True, dropout=dropout)  # 双向GRU
        self.attention = SelfAttention(hidden_dim * 2)  # 自注意力机制
        self.fc = nn.Linear(hidden_dim * 2, num_class)  # 全连接层
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, x):  # x: [B, T]
        emb = self.embedding(x)  # [B, T, E]
        gru_out, _ = self.gru(emb)  # lstm_out: [B, T, 2H]
        # attended, attention_weights = self.attention(gru_out)  # [B, 2H], [B, T, 1]
        pooled = torch.mean(gru_out, dim=1)  # [B, 2H]
        out = self.dropout(pooled)
        return self.fc(out), 0#attention_weights  # logits, attention_weights
        # 前向传播
        # 输入：词表大小，嵌入维度，隐藏维度，类别数
        # 输出：logits, attention_weights
        # logits：分类结果
        # attention_weights：注意力权重