import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_loader import get_dataloaders

def compute_class_weights(data_dir):
    """计算类别权重"""
    train_loader, _, _, _ = get_dataloaders(data_dir, batch_size=32, max_len=128)
    
    # 收集所有标签
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    
    print(f"类别分布: {np.bincount(all_labels)}")
    print(f"类别权重: {class_weights}")
    print(f"标签0(消极)权重: {class_weights[0]:.4f}")
    print(f"标签1(积极)权重: {class_weights[1]:.4f}")
    
    return torch.FloatTensor(class_weights)

if __name__ == "__main__":
    weights = compute_class_weights("data")
    print(f"\n建议在训练时使用: criterion = nn.CrossEntropyLoss(weight={weights.tolist()})")