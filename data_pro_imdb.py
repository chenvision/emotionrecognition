import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# 1. 读取IMDB CSV文件
df = pd.read_csv("./IMDB Dataset.csv")  # 确保路径正确

print(df.head())

# 2. 标签分布
label_counts = df["sentiment"].value_counts()
print(label_counts)

# label_counts.plot(kind="bar", title="Sentiment Distribution", rot=0)
# plt.ylabel("Number of Reviews")
# plt.xlabel("Sentiment")
# plt.tight_layout()
# plt.show()

# 3. 标签映射（positive -> 1, negative -> 0）
label_map = {"positive": 1, "negative": 0}
df["label"] = df["sentiment"].map(label_map)


# 先清洗评论内容
# def clean_review(text):
#     text = text.lower()
#     text = re.sub(r"<.*?>", " ", text)  # 去除HTML标签
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

def clean_review(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["review"] = df["review"].astype(str).apply(clean_review)

# 4. 检查是否有未识别标签
print("Unmapped labels:", df["label"].isna().sum())

# 5. 句子长度分布
df["length"] = df["review"].apply(lambda x: len(str(x)))

# plt.hist(df["length"], bins=30, color="lightgreen")
# plt.title("Review Length Distribution")
# plt.xlabel("Number of Characters")
# plt.ylabel("Number of Reviews")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

print(df["length"].describe())

# # 6. 生成正面评论词云
# import jieba  # 防止中文数据（你可以删除jieba，如果全是英文）
# all_positive = " ".join(df[df["label"] == 1]["review"].tolist())
# wordcloud = WordCloud(
#     background_color="white",
#     width=800,
#     height=400
# ).generate(all_positive)
#
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.title("Word Cloud of Positive Reviews")
# plt.tight_layout()
# plt.show()

from sklearn.model_selection import train_test_split

# 打乱数据
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 划分train/val/test
train_val, test = train_test_split(df_shuffled, test_size=0.1, random_state=42, stratify=df_shuffled["label"])
train, val = train_test_split(train_val, test_size=1/9, random_state=42, stratify=train_val["label"])

print(f"Train size: {len(train)}")
print(f"Val size: {len(val)}")
print(f"Test size: {len(test)}")

# 保存
train.to_csv("data/imdb_train.csv", index=False)
val.to_csv("data/imdb_val.csv", index=False)
test.to_csv("data/imdb_test.csv", index=False)

# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# token_lens = df["review"].apply(lambda x: len(tokenizer.tokenize(x)))
#
# token_lens.hist(bins=50)
# plt.title("Token Length Distribution")
# plt.xlabel("Token Count")
# plt.ylabel("Number of Samples")
# plt.grid(True)
# plt.show()
#
# # 字符长度直方图 + 分位线
# plt.figure(figsize=(8, 5))
# plt.hist(df["length"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
#
# for q in [0.5, 0.75, 0.9, 0.95]:
#     v = df["length"].quantile(q)
#     plt.axvline(v, linestyle='--', label=f'{int(q * 100)}% = {int(v)}')
#
# plt.title("IMDB Review Length Distribution with Quantiles")
# plt.xlabel("Character Count")
# plt.ylabel("Number of Reviews")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 统计每个数据集的标签分布
train_counts = train["label"].value_counts().sort_index()
val_counts = val["label"].value_counts().sort_index()
test_counts = test["label"].value_counts().sort_index()

# 标签（x轴）
labels = ["Negative", "Positive"]
x = np.arange(len(labels))  # [0, 1]

# 柱状图的宽度
bar_width = 0.25

# 创建图像
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width, train_counts, width=bar_width, label="Train", color="#4CAF50")
plt.bar(x, val_counts, width=bar_width, label="Validation", color="#2196F3")
plt.bar(x + bar_width, test_counts, width=bar_width, label="Test", color="#FFC107")

# 添加标签和标题
plt.xticks(x, labels)
plt.ylabel("Number of Samples")
plt.title("Label Distribution Across Train/Val/Test")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()