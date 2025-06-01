"""TextCNN classifier."""
from __future__ import annotations
import torch
from torch import nn

__all__ = ["TextCNNClassifier"]


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_class: int = 2, kernel_sizes=(3, 4, 5), num_channels=100, dropout=0.5, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)# 嵌入层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes]
        )# 卷积层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_class)
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    def forward(self, x):  # x: [B, T]
        emb = self.embedding(x).unsqueeze(1)  # [B, 1, T, E]
        conv_outs = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(out, dim=2)[0] for out in conv_outs]
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))