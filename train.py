"""Generic training loop – select model via CLI."""
from __future__ import annotations
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import io
from PIL import Image

from data_loader import get_dataloaders
from utils import compute_metrics, DEVICE
from model_lstm import BiLSTMClassifier
from model_textcnn import TextCNNClassifier
from model_gru import BiGRUClassifier

# ----------------------------------------------------------------------
def plot_attention_map(attention_weights, writer, epoch):
    # 仅展示前几个样本
    for i in range(min(3, attention_weights.shape[0])):
        attn = attention_weights[i].squeeze(-1).cpu().numpy()  # [T]
        fig, ax = plt.subplots(figsize=(10, 1))
        ax.imshow(attn[np.newaxis, :], cmap="viridis", aspect="auto")
        ax.set_title(f"Attention Weights Sample {i}")
        writer.add_figure(f"Attention/sample_{i}", fig, global_step=epoch)
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
    return metrics, labels, preds


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", choices=["lstm", "textcnn", "gru"], default="gru")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=999)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        args.data_dir, args.batch_size, args.max_len
    )
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    log_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{args.model}/{log_tag}")
    if args.model == "lstm":
        model_cls = BiLSTMClassifier
    elif args.model == "textcnn":
        model_cls = TextCNNClassifier
    elif args.model == "gru":
        model_cls = BiGRUClassifier
    else:
        raise ValueError('Model must be either "lstm" or "textcnn" or "gru"')
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
        val_metrics, _, _ = evaluate(model, val_loader, criterion)
        
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # 更新学习率
        scheduler.step(val_metrics['f1'])
        # 记录可视化节点
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Precision/val", val_metrics["precision"], epoch)
        writer.add_scalar("Recall/val", val_metrics["recall"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
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
            plot_attention_map(val_attention, writer, epoch)
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs without improvement")
                break

    # Final test evaluation
    ckpt = torch.load(Path(args.save_path) / f"best_{args.model}.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_metrics, labels, preds = evaluate(model, test_loader, criterion)
    print(
        f"Test – loss: {test_metrics['loss']:.4f}, "
        f"acc: {test_metrics['accuracy']:.4f}, f1: {test_metrics['f1']:.4f}"
    )

    # ---- 分类报告到 TensorBoard ----
    report_dict = classification_report(labels, preds, target_names=["Negative", "Positive"], output_dict=True)
    for label in ["Negative", "Positive"]:
        writer.add_scalar(f"Test/Precision_{label}", report_dict[label]["precision"], 0)
        writer.add_scalar(f"Test/Recall_{label}", report_dict[label]["recall"], 0)
        writer.add_scalar(f"Test/F1_{label}", report_dict[label]["f1-score"], 0)

    writer.add_scalar("Test/Accuracy", report_dict["accuracy"], 0)
    writer.add_scalar("Test/Macro_F1", report_dict["macro avg"]["f1-score"], 0)

    # ---- 混淆矩阵图像写入 TensorBoard ----
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # 将图像写入 TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    writer.add_image("Test/Confusion_Matrix", image, 0, dataformats='HWC')
    buf.close()
    plt.close(fig)
    writer.close()


if __name__ == "__main__":
    main()
