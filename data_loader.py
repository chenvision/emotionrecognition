from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from utils import tokenize

__all__ = ["get_dataloaders", "TextSentimentDataset"]



class TextSentimentDataset(Dataset):
    """Load a TSV with columns <text>\t<label> where label is 0/1."""

    def __init__(self, file_path: Path, vocab: Dict[str, int] | None = None, max_len: int = 128):# 读取数据
        self.samples: List[Tuple[str, int]] = []# 样本列表
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                txt, lbl = parts[0], int(parts[1])
                self.samples.append((txt, lbl))#
        self.max_len = max_len
        self.vocab = vocab if vocab is not None else self._build_vocab()# 构建词表


    # ------------------------------------------------------------------
    def _build_vocab(self) -> Dict[str, int]:# 构建词表
        counter = Counter()
        for txt, _ in self.samples:
            counter.update(tokenize(txt))
        vocab = {"<pad>": 0, "<unk>": 1}
        for tok, _ in counter.most_common():
            vocab[tok] = len(vocab)
        return vocab

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[int]:# 编码文本
        ids = [self.vocab.get(t, 1) for t in tokenize(text)][: self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return ids

    # ------------------------------------------------------------------
    def __len__(self) -> int:# 返回样本数
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):# 返回样本
        text, label = self.samples[idx]
        return torch.tensor(self._encode(text), dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ----------------------------------------------------------------------

def get_dataloaders(data_dir: str | Path, batch_size: int = 32, max_len: int = 128):# 返回数据加载器
    data_dir = Path(data_dir)
    train_ds = TextSentimentDataset(data_dir / "train.tsv", max_len=max_len)
    val_ds = TextSentimentDataset(data_dir / "dev.tsv", vocab=train_ds.vocab, max_len=max_len)
    test_ds = TextSentimentDataset(data_dir / "test.tsv", vocab=train_ds.vocab, max_len=max_len)

    def _create_loader(ds):# 创建数据加载器
        return DataLoader(ds, batch_size=batch_size, shuffle=ds is train_ds)

    return _create_loader(train_ds), _create_loader(val_ds), _create_loader(test_ds), train_ds.vocab