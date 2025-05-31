from __future__ import annotations
import re
import jieba
from typing import List
import torch
from sklearn.metrics import accuracy_score, f1_score

__all__ = ["tokenize", "compute_metrics", "DEVICE"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fa5]")


def _is_chinese(text: str) -> bool:
    return bool(_CHINESE_CHAR_PATTERN.search(text))


def tokenize(text: str) -> List[str]:
    """Crude multilingual tokenizer – uses jieba for Chinese, whitespace for others."""
    if _is_chinese(text):
        return [tok for tok in jieba.lcut(text) if tok.strip()]
    return text.strip().lower().split()


def compute_metrics(preds: List[int], labels: List[int]) -> dict[str, float]:
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}