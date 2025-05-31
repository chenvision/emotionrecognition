"""Generic training loop – select model via CLI."""
from __future__ import annotations
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path

from data_loader import get_dataloaders
from utils import compute_metrics, DEVICE
from model_lstm import BiLSTMClassifier
from model_textcnn import TextCNNClassifier

# ----------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)  # 忽略训练时的attention_weights
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    attention_weights_list = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, attention_weights = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        attention_weights_list.append(attention_weights.cpu())
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["attention_weights"] = torch.cat(attention_weights_list, dim=0)
    return metrics


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", choices=["lstm", "textcnn"], default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        args.data_dir, args.batch_size, args.max_len
    )
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    model_cls = BiLSTMClassifier if args.model == "lstm" else TextCNNClassifier
    model = model_cls(len(vocab)).to(DEVICE)

    # 计算类别权重以解决类别不平衡问题
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # 收集训练数据的所有标签
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    # 计算平衡权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"使用类别权重: 消极={class_weights[0]:.4f}, 积极={class_weights[1]:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    best_f1 = 0.0
    no_improve_count = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)
        
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # 更新学习率
        scheduler.step(val_metrics['f1'])
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            no_improve_count = 0
            # 保存模型时同时保存注意力权重
            val_attention = val_metrics.pop("attention_weights")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "args": vars(args),
                    "attention_weights": val_attention,
                    "best_f1": best_f1,
                    "epoch": epoch,
                },
                Path(args.save_path) / f"best_{args.model}.pt",
            )
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs without improvement")
                break

    # Final test evaluation
    ckpt = torch.load(Path(args.save_path) / f"best_{args.model}.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion)
    print(
        f"Test – loss: {test_metrics['loss']:.4f}, "
        f"acc: {test_metrics['accuracy']:.4f}, f1: {test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
