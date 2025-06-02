import torch
from torch import nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-chinese", num_class: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_class)

    def forward(self, input_ids, attention_mask=None):
        # [B, T] → outputs: BaseModelOutputWithPooling
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [B, H]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_class]
        return logits  # 与 RNN 系列保持一致
