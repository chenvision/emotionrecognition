import torch
import torch.nn as nn
class BaggingEnsemble(nn.Module):
    def __init__(self, base_model_cls, model_args, num_models=5):
        super().__init__()
        self.models = nn.ModuleList([
            base_model_cls(**model_args) for _ in range(num_models)
        ])
        self.num_models = num_models

    def forward(self, x):
        logits_list = []
        attention_list = []

        for model in self.models:
            logits, attention = model(x)
            logits_list.append(logits)
            attention_list.append(attention)

        # 将多个模型输出的logits取平均（也可使用投票）
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)

        return avg_logits, attention_list
