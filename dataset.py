from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer

class EmotionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 将文本标签映射为整数：positive → 1, negative → 0
        self.label_map = {"positive": 1, "negative": 0}
        self.data = self.data[self.data["label"].isin(self.label_map)]  # 清理未知标签
        self.data["label"] = self.data["label"].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
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
