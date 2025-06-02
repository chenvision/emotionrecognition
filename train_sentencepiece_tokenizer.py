import pandas as pd
import sentencepiece as spm
import os
from transformers import PreTrainedTokenizerFast

# === Step 1: 提取文本数据，保存为纯文本格式（每行一句） ===
def extract_corpus(csv_path: str, output_txt_path: str):
    df = pd.read_csv(csv_path)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for line in df["text"]:
            line = str(line).strip().replace("\n", "")
            if line:
                f.write(line + "\n")
    print(f"[✓] 文本已保存至: {output_txt_path}")

# === Step 2: 训练 SentencePiece Tokenizer ===
def train_sentencepiece(input_txt: str, model_prefix: str = "spm_emotion", vocab_size: int = 16000):
    spm.SentencePieceTrainer.train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",  # 可选：bpe / unigram / char / word
        character_coverage=0.9995,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3
    )
    print(f"[✓] Tokenizer 模型已训练完成: {model_prefix}.model / {model_prefix}.vocab")

# === Step 3: 测试 tokenizer 的效果 ===
def test_tokenizer(model_path: str, test_text: str = "我真的好喜欢今天的天气！"):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    tokens = sp.encode(test_text, out_type=str)
    ids = sp.encode(test_text, out_type=int)

    print("\n[示例测试]")
    print(f"原始文本: {test_text}")
    print(f"分词结果: {tokens}")
    print(f"对应 ID: {ids}")
    print(f"解码还原: {sp.decode(ids)}")

# === 主流程 ===
if __name__ == "__main__":
    # 文件路径
    csv_path = "backup/data/emotion_train.csv"
    corpus_txt = "emotion_corpus.txt"
    model_prefix = "spm_emotion"
    vocab_size = 16000

    # 执行流程
    extract_corpus(csv_path, corpus_txt)
    train_sentencepiece(corpus_txt, model_prefix, vocab_size)
    test_tokenizer(f"{model_prefix}.model")

    # ✅ 正确方式：使用 PreTrainedTokenizerFast 包装 .model（本地）
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="spm_emotion.model",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>"
    )

    # ✅ 保存为 HuggingFace 兼容格式
    save_path = os.path.abspath("./tokenizer_dir")
    tokenizer.save_pretrained(save_path)
    print(f"[✓] Tokenizer 已成功保存至: {save_path}")
