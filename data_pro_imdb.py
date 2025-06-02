import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
# 1. 读取IMDB CSV文件
df = pd.read_csv("./IMDB Dataset.csv")  # 确保路径正确

print(df.head())

# 2. 标签分布
label_counts = df["sentiment"].value_counts()
print(label_counts)

label_counts.plot(kind="bar", title="Sentiment Distribution", rot=0)
plt.ylabel("Number of Reviews")
plt.xlabel("Sentiment")
plt.tight_layout()
plt.show()

# 3. 标签映射（positive -> 1, negative -> 0）
label_map = {"positive": 1, "negative": 0}
df["label"] = df["sentiment"].map(label_map)


# 先清洗评论内容
def clean_review(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # 去除HTML标签
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["review"] = df["review"].astype(str).apply(clean_review)

# 4. 检查是否有未识别标签
print("Unmapped labels:", df["label"].isna().sum())

# 5. 句子长度分布
df["length"] = df["review"].apply(lambda x: len(str(x)))

plt.hist(df["length"], bins=30, color="lightgreen")
plt.title("Review Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Reviews")
plt.grid(True)
plt.tight_layout()
plt.show()

print(df["length"].describe())

# 6. 生成正面评论词云
import jieba  # 防止中文数据（你可以删除jieba，如果全是英文）
all_positive = " ".join(df[df["label"] == 1]["review"].tolist())
wordcloud = WordCloud(
    background_color="white",
    width=800,
    height=400
).generate(all_positive)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Positive Reviews")
plt.tight_layout()
plt.show()

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
