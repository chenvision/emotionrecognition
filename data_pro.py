import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

# 1. Load and parse the XML file
tree = ET.parse("./evtestdata1/Training data for Emotion Classification.xml")
root = tree.getroot()

# 2. Extract emotion-labeled sentences
data = []
for weibo in root.findall("weibo"):
    for sentence in weibo.findall("sentence"):
        text = sentence.text.strip()
        if sentence.get("opinionated") == "Y":
            emotion = sentence.get("emotion-1-type")
            if emotion and emotion != "none":
                data.append({"text": text, "emotion": emotion})

df = pd.DataFrame(data)
print(df.head())

# 3. Emotion category distribution
emotion_counts = df["emotion"].value_counts()
print(emotion_counts)

emotion_counts.plot(kind="bar", title="Emotion Category Distribution", rot=45)
plt.ylabel("Number of Sentences")
plt.xlabel("Emotion Type")
plt.tight_layout()
plt.show()

# 4. Map to binary labels (positive / negative)
def map_emotion(e):
    if e in {"happiness", "like", "surprise"}:
        return "positive"
    elif e in {"sadness", "anger", "disgust", "fear"}:
        return "negative"
    else:
        return None

df["label"] = df["emotion"].apply(map_emotion)

label_counts = df["label"].value_counts()
print(label_counts)

label_counts.plot(kind="pie", autopct="%1.1f%%", title="Positive vs Negative Distribution", ylabel="")
plt.tight_layout()
plt.show()

# 5. Sentence length distribution
df["length"] = df["text"].apply(len)

plt.hist(df["length"], bins=30, color="skyblue")
plt.title("Sentence Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Sentences")
plt.grid(True)
plt.tight_layout()
plt.show()

print(df["length"].describe())

# 6. Word cloud for positive samples
all_positive = " ".join(df[df["label"] == "positive"]["text"].tolist())
positive_words = " ".join(jieba.cut(all_positive))

wordcloud = WordCloud(
    font_path="simhei.ttf",
    background_color="white",
    width=800,
    height=400
).generate(positive_words)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Positive Emotion")
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

# 保证打乱
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 先拆出 test（10%），剩下90%
train_val, test = train_test_split(df_shuffled, test_size=0.1, random_state=42, stratify=df_shuffled["label"])

# 再拆出 val（从 train_val 中取 1/9，相当于总数的10%）
train, val = train_test_split(train_val, test_size=1/9, random_state=42, stratify=train_val["label"])

# 检查数量
print(f"Train size: {len(train)}")
print(f"Val size: {len(val)}")
print(f"Test size: {len(test)}")

# 保存为 CSV
train.to_csv("data/emotion_train.csv", index=False)
val.to_csv("data/emotion_val.csv", index=False)
test.to_csv("data/emotion_test.csv", index=False)
