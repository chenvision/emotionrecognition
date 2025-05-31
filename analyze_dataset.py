from collections import Counter
import matplotlib.pyplot as plt
from data_loader import TextSentimentDataset
from utils import tokenize
from wordcloud import WordCloud

def analyze_label_distribution(dataset: TextSentimentDataset, title="Label Distribution"):
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    print(f"Label count: {dict(counter)}")

    plt.figure(figsize=(5, 4))
    plt.bar(["Negative (0)", "Positive (1)"], [counter[0], counter[1]], color=['red', 'green'])
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()


def analyze_length_distribution(dataset: TextSentimentDataset):
    lengths = [len(tokenize(text)) for text, _ in dataset.samples]
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title("Token Length Distribution")
    plt.xlabel("Number of tokens per sentence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    avg_len = sum(lengths) / len(lengths)
    print(f"Average token length: {avg_len:.2f}")


def show_wordcloud(dataset: TextSentimentDataset, label_filter=None):
    counter = Counter()
    for text, label in dataset.samples:
        if label_filter is None or label == label_filter:
            counter.update(tokenize(text))
    wordcloud = WordCloud(font_path='simhei.ttf', width=800, height=400, background_color='white').generate_from_frequencies(counter)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    title = f"WordCloud - {'All' if label_filter is None else ('Positive' if label_filter==1 else 'Negative')}"
    plt.title(title)
    plt.show()

def compare_length_by_label(dataset: TextSentimentDataset):
    pos_lengths = [len(tokenize(text)) for text, label in dataset.samples if label == 1]
    neg_lengths = [len(tokenize(text)) for text, label in dataset.samples if label == 0]
    print(f"Avg positive length: {sum(pos_lengths)/len(pos_lengths):.2f}")
    print(f"Avg negative length: {sum(neg_lengths)/len(neg_lengths):.2f}")
    plt.boxplot([neg_lengths, pos_lengths], tick_labels=["Negative", "Positive"])
    plt.title("Token Length by Sentiment")
    plt.ylabel("Number of Tokens")
    plt.grid(True)
    plt.show()

def print_sample(dataset: TextSentimentDataset, n=5):
    for i in range(n):
        text, label = dataset.samples[i]
        print(f"【Label】{'Positive' if label == 1 else 'Negative'}\n{text}\n")

if __name__ == "__main__":
    # 1. 加载训练集
    dataset = TextSentimentDataset("data/train.tsv", max_len=128)

    # 2. 执行分析函数
    analyze_label_distribution(dataset)
    analyze_length_distribution(dataset)
    compare_length_by_label(dataset)
    show_wordcloud(dataset)  # 所有词
    show_wordcloud(dataset, label_filter=0)  # 消极
    show_wordcloud(dataset, label_filter=1)  # 积极
    print_sample(dataset, n=5)