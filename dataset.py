from torch.utils.data import Dataset
import pandas as pd
import torch
import types

class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64, is_imdb=False):
        """
        通用情感分类数据集:
        - 中文数据: text + label
        - IMDB数据: review + sentiment (需要映射成label)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_imdb = is_imdb

        if self.is_imdb:
            # ✅ 1. IMDB数据 sentiment -> label (positive:1, negative:0)
            self.label_map = {"positive": 1, "negative": 0}
            self.data["label"] = self.data["sentiment"].map(self.label_map)
        else:
            # ✅ 2. 中文情感分类
            self.label_map = {"positive": 1, "negative": 0}
            self.data["label"] = self.data["label"].map(lambda x: self.label_map.get(x, -1))
            # 过滤非法label
            self.data = self.data[self.data["label"] != -1]

        # 判定 tokenizer 风格
        self.use_hf_style = callable(getattr(tokenizer, "__call__", None))  # HuggingFace tokenizer
        self.use_spm_style = isinstance(getattr(tokenizer, "encode", None), types.MethodType)  # SentencePiece tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_imdb:
            text = self.data.iloc[idx]["review"]  # ✅ IMDB取'review'
        else:
            text = self.data.iloc[idx]["text"]    # ✅ 中文取'text'

        label = self.data.iloc[idx]["label"]      # ✅ label统一是数字 0/1

        if self.use_hf_style:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        elif self.use_spm_style:
            ids = self.tokenizer.encode(text, out_type=int)
            ids = ids[:self.max_length]
            pad_id = 0
            input_ids = ids + [pad_id] * (self.max_length - len(ids))
            attention_mask = [1 if i < len(ids) else 0 for i in range(self.max_length)]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            raise ValueError("Unsupported tokenizer type")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }



def get_tokenizer(model_name="bert-base-chinese"):
    """
    Load and return a tokenizer by model name.

    Parameters:
        model_name (str): Name of pretrained model, e.g. 'bert-base-chinese' or 'xlm-roberta-base'

    Returns:
        tokenizer (transformers.PreTrainedTokenizer): Loaded tokenizer instance
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[INFO] Loaded tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer '{model_name}': {e}")
        raise
