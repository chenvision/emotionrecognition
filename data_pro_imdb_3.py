import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
from itertools import chain, product
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from model_lstm import BiLSTMClassifier  # 🔺确保你实现了这个类
from model_gru import BiGRUClassifier
nltk.download('punkt')
nltk.download('stopwords')

# ========================== Dataset 相关 ==========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

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
    labels = torch.tensor(label_list, dtype=torch.long)
    return padded[:, :512], labels

# ========================== 评估函数 ==========================

def evaluate(model, dataloader, criterion, device, show_report=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
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
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    return total_loss / len(dataloader), acc, precision, recall, f1

# ========================== 主程序入口 ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================== 加载数据 ==========================

    df = pd.read_csv("IMDB Dataset.csv").drop_duplicates()
    df["tokens"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

    # 词汇表构建
    all_tokens = list(chain.from_iterable(df["tokens"]))
    word_counts = Counter(all_tokens)
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    def encode_tokens(tokens):
        return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]

    df["input_ids"] = df["tokens"].apply(encode_tokens)

    # 数据划分
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df['input_ids'], df['label'], test_size=0.1, stratify=df['label'], random_state=42
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=1/9, stratify=train_val_labels, random_state=42
    )

    # 构造 Dataset 和 DataLoader
    train_loader = DataLoader(IMDBDataset(train_texts.tolist(), train_labels.tolist()),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(IMDBDataset(val_texts.tolist(), val_labels.tolist()),
                            batch_size=args.batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(IMDBDataset(test_texts.tolist(), test_labels.tolist()),
                             batch_size=args.batch_size, collate_fn=collate_batch)

    vocab_size = len(word2idx)
    criterion = nn.CrossEntropyLoss()

    # ========================== Grid Search 超参数组合 ==========================

    param_grid = {
        "embed_dim": [64, 128, 256],
        "hidden_dim": [64, 128, 256],
        "num_layers": [1, 2]
    }
    param_combinations = list(product(param_grid["embed_dim"],
                                      param_grid["hidden_dim"],
                                      param_grid["num_layers"]))

    results = []
    cnt = 0
    for embed_dim, hidden_dim, num_layers in param_combinations:
        # if cnt <= 4:
        #     cnt += 1
        #     continue
        print(f"\n--- Training GRU | embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers} ---")
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_class=2,
            dropout=0.3,
            num_layers=num_layers,
            use_attention=False
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_f1 = 0.0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            correct, total = 0, 0

            for batch in train_loader:
                input_ids, labels = batch[0].to(device), batch[1].to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            val_loss, val_acc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)
            best_val_f1 = max(best_val_f1, f1)
            print(f"[Epoch {epoch+1}] TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, F1={f1:.4f}")

        # 评估测试集
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            model, test_loader, criterion, device, show_report=False
        )

        config_name = f"embed{embed_dim}-hid{hidden_dim}-lay{num_layers}"

        results.append({
            "config": config_name,  # 新增列
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "val_f1": best_val_f1,
            "val_acc": val_acc,
            "val_precision": prec,
            "val_recall": rec,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec
        })

    # 保存结果
    df_result = pd.DataFrame(results).sort_values(by="val_f1", ascending=False)
    cols = ["config", "embed_dim", "hidden_dim", "num_layers",
            "val_f1", "val_acc", "val_precision", "val_recall",
            "test_f1", "test_acc", "test_precision", "test_recall"]
    df_result = df_result[cols]

    df_result.to_csv("grid_search_lstm_results_with_test_gru.csv", index=False)
    print("\n✅ Grid Search Completed. Top Results:")
    print(df_result.head())

