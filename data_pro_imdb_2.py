import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import binary_cross_entropy_with_logits
import argparse
import torch

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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter
from itertools import chain

nltk.download('punkt')
nltk.download('stopwords')

def evaluate(model, dataloader, criterion, device, show_report=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
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
    parser.add_argument("--model_type", type=str, choices=["lstm", "gru", "textcnn", "bert", "bagging"], default="textcnn")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # 编码器
    parser.add_argument("--model_name", type=str, default="bert-base-uncased") # bert-base-uncased
    parser.add_argument("--max_length", type=int, default=512)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="./weights/model_rnn.pt")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    df = pd.read_csv('IMDB Dataset.csv')
    df.drop_duplicates(inplace=True)

    # 文本清洗与分词
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    df['tokens'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    # 构建词汇表（不使用 torchtext）
    all_tokens = list(chain.from_iterable(df['tokens']))
    word_counts = Counter(all_tokens)
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1

    # 编码为索引
    def encode_tokens(tokens):
        return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

    df['input_ids'] = df['tokens'].apply(encode_tokens)

    # 划分数据集
    # X_train, X_test, y_train, y_test = train_test_split(df['input_ids'], df['label'], test_size=0.2, random_state=42)

    # 第一步：先划分出临时训练集和测试集（90%训练 + 10%测试）
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df['input_ids'], df['label'], test_size=0.1, random_state=42, stratify=df['label']
    )

    # 第二步：再从训练集中划出验证集（从90%中划出10%，即0.1 / 0.9 ≈ 11.1%）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=1 / 9, random_state=42, stratify=train_val_labels
    )

    # PyTorch Dataset 和 DataLoader
    class IMDBDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    def collate_batch(batch):
        text_list, label_list = zip(*batch)
        padded = pad_sequence(text_list, batch_first=True, padding_value=word2idx['<PAD>'])
        labels = torch.tensor(label_list, dtype=torch.long)  # 👈 转为 Long 且保持 1D
        return padded[:, :args.max_length], labels

    # 构造 Dataset
    train_dataset = IMDBDataset(train_texts.tolist(), train_labels.tolist())
    val_dataset = IMDBDataset(val_texts.tolist(), val_labels.tolist())
    test_dataset = IMDBDataset(test_texts.tolist(), test_labels.tolist())

    # 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    vocab_size = len(word2idx)
    # 模型选择
    if args.model_type == "lstm":
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_class=2,
            dropout=args.dropout,
            num_layers=1,
            use_attention=False
        )
    elif args.model_type == "gru":
        model = BiGRUClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_class=2,
            dropout=args.dropout
        )
    elif args.model_type == "textcnn":
        model = TextCNNClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_class=2,
            dropout=args.dropout
        )
    elif args.model_type == "bagging":
        model = BaggingEnsemble(
            base_model_cls=BiLSTMClassifier,
            num_models=5,
            model_args={
                "vocab_size": vocab_size,
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
            input_ids = batch[0].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            labels = batch[1].to(device)

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


if __name__ == "__main__":
    main()
