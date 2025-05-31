import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from predict import load_model, encode, tokenize
from model_lstm import BiLSTMClassifier

def quick_test(model, vocab, max_len: int, text: str):
    x = encode(text, vocab, max_len)
    model.eval()
    with torch.no_grad():
        if isinstance(model, BiLSTMClassifier):
            logits, _ = model(x)
        else:
            logits = model(x)
            
        probs = F.softmax(logits, dim=-1).squeeze()
        label = int(torch.argmax(logits))
        confidence = probs[label].item()
        
        print(f"'{text}' -> {('积极' if label == 1 else '消极')} ({confidence:.3f})")

if __name__ == "__main__":
    model, vocab, max_len = load_model("checkpoints/best_lstm.pt")
    
    # 测试消极文本
    negative_texts = [
        "今天天气真差",
        "我真服了，逆天", 
        "我操你妈",
        "废物",
        "垃圾产品",
        "太糟糕了",
        "我很生气",
        "讨厌死了"
    ]
    
    print("=== 消极文本测试 ===")
    for text in negative_texts:
        quick_test(model, vocab, max_len, text)
    
    print("\n=== 积极文本测试 ===")
    positive_texts = ["这个产品非常好用", "我很开心", "太棒了"]
    for text in positive_texts:
        quick_test(model, vocab, max_len, text)