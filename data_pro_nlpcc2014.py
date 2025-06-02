import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
from sklearn.model_selection import train_test_split

# ========== 配置 ==========
USE_WORD_SEGMENTATION = False  # True 则启用 jieba 分词 + 停用词去除，适用于 RNN/TextCNN 等
STOPWORDS_PATH = "hit_stopwords.txt"  # 停用词文件路径
# ==========================

# 加载 XML 数据
tree = ET.parse("./evtestdata1/Training data for Emotion Classification.xml")
root = tree.getroot()

# 解析标注情绪句子
data = []
for weibo in root.findall("weibo"):
    for sentence in weibo.findall("sentence"):
        text = sentence.text.strip()
        if sentence.get("opinionated") == "Y":
            emotion = sentence.get("emotion-1-type")
            if emotion and emotion != "none":
                data.append({"text": text, "emotion": emotion})

df = pd.DataFrame(data)

# 映射为二分类标签
def map_emotion(e):
    if e in {"happiness", "like", "surprise"}:
        return "positive"
    elif e in {"sadness", "anger", "disgust", "fear"}:
        return "negative"
    else:
        return None

df["label"] = df["emotion"].apply(map_emotion)
df = df[df["label"].notna()].reset_index(drop=True)

# 可选：分词 + 停用词去除（适用于 RNN/TextCNN）
if USE_WORD_SEGMENTATION:
    def load_stopwords(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    stopwords = load_stopwords(STOPWORDS_PATH)

    def clean_text(text):
        tokens = jieba.lcut(text)
        return " ".join([w for w in tokens if w not in stopwords and w.strip()])

    df["text"] = df["text"].apply(clean_text)

# 打乱并划分 train/val/test（8:1:1）
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_val, test = train_test_split(df_shuffled, test_size=0.1, stratify=df_shuffled["label"], random_state=42)
train, val = train_test_split(train_val, test_size=1/9, stratify=train_val["label"], random_state=42)

print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

# 保存为 CSV（适用于训练）
train.to_csv("data/emotion_train.csv", index=False)
val.to_csv("data/emotion_val.csv", index=False)
test.to_csv("data/emotion_test.csv", index=False)

# 可视化统计图
df["length"] = df["text"].apply(len)
df["emotion"].value_counts().plot(kind="bar", title="Emotion Category Distribution", rot=45)
plt.ylabel("Number of Sentences")
plt.tight_layout()
plt.show()

df["label"].value_counts().plot(kind="pie", autopct="%1.1f%%", title="Positive vs Negative", ylabel="")
plt.tight_layout()
plt.show()

plt.hist(df["length"], bins=30, color="skyblue")
plt.title("Sentence Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Sentences")
plt.grid(True)
plt.tight_layout()
plt.show()

# WordCloud 正向情绪词云
all_positive = " ".join(df[df["label"] == "positive"]["text"].tolist())
positive_words = jieba.lcut(all_positive) if not USE_WORD_SEGMENTATION else all_positive.split()

wordcloud = WordCloud(font_path="simhei.ttf", background_color="white", width=800, height=400).generate(" ".join(positive_words))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Positive Emotion")
plt.tight_layout()
plt.show()
