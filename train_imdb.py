import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from dataset import EmotionDataset
import matplotlib.pyplot as plt
import seaborn as sns
from model_gru import BiGRUClassifier
from model_lstm import BiLSTMClassifier
from model_textcnn import TextCNNClassifier
from model_bert import BertClassifier
from model_bagging import BaggingEnsemble
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, dataloader, criterion, device, show_report=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs, _ = model(input_ids)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    if show_report:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4))
        print("Confusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    return total_loss / len(dataloader), acc, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Train BiRNN (LSTM/GRU) for emotion classification")

    # 数据路径
    parser.add_argument("--dataset", choices=["chinese", "imdb"], default="imdb", help="选择数据集：chinese 或 imdb")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)

    # 模型结构参数
    parser.add_argument("--model_type", type=str, choices=["lstm", "gru", "textcnn", "bert", "bagging"], default="lstm")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # 编码器
    parser.add_argument("--model_name", type=str, default="bert-base-uncased") # bert-base-uncased
    parser.add_argument("--max_length", type=int, default=64)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="./weights/model_rnn.pt")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 根据数据集选择加载方式
    if args.dataset == "chinese":
        train_path = args.train_path or "./data/emotion_train.csv"
        val_path = args.val_path or "./data/emotion_val.csv"
        test_path = args.test_path or "./data/emotion_test.csv"

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        train_dataset = EmotionDataset(train_df, tokenizer, args.max_length, is_imdb=False)
        val_dataset = EmotionDataset(val_df, tokenizer, args.max_length, is_imdb=False)
        test_dataset = EmotionDataset(test_df, tokenizer, args.max_length, is_imdb=False)

    elif args.dataset == "imdb":
        train_path = args.train_path or "./data/imdb_train.csv"
        val_path = args.val_path or "./data/imdb_val.csv"
        test_path = args.test_path or "./data/imdb_test.csv"

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        train_dataset = EmotionDataset(train_df, tokenizer, args.max_length, is_imdb=True)
        val_dataset = EmotionDataset(val_df, tokenizer, args.max_length, is_imdb=True)
        test_dataset = EmotionDataset(test_df, tokenizer, args.max_length, is_imdb=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 模型选择
    if args.model_type == "lstm":
        model = BiLSTMClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_class=2,
            dropout=args.dropout,
            num_layers=1,
            use_attention=True
        )
    elif args.model_type == "gru":
        model = BiGRUClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_class=2,
            dropout=args.dropout
        )
    elif args.model_type == "textcnn":
        model = TextCNNClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            num_class=2,
            dropout=args.dropout
        )
    elif args.model_type == "bert":
        model = BertClassifier(model_name=args.model_name, num_class=2)
    elif args.model_type == "bagging":
        model = BaggingEnsemble(
            base_model_cls=BiLSTMClassifier,
            num_models=5,
            model_args={
                "vocab_size": tokenizer.vocab_size,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_class": 2,
                "dropout": 0.1,
                "num_layers": 1,
                "use_attention": True
            }
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    writer = SummaryWriter(log_dir=f"runs/{args.model_type}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if args.model_type == "bert":
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits, _ = model(input_ids)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # 验证
        val_loss, val_acc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Accuracy", val_acc, epoch)
        writer.add_scalar("Metrics/Precision", prec, epoch)
        writer.add_scalar("Metrics/Recall", rec, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)

        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")

    # 绘制曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.show()

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    print("\n[Test Evaluation]")
    evaluate(model, test_loader, criterion, device, show_report=True)
    writer.close()

if __name__ == "__main__":
    main()
