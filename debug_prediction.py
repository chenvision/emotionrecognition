import torch
import torch.nn.functional as F
from predict import load_model, encode, tokenize
from model_lstm import BiLSTMClassifier

def debug_predict(model, vocab, max_len: int, text: str):
    print(f"输入文本: {text}")
    
    # 编码
    x = encode(text, vocab, max_len)
    tokens = tokenize(text)[:max_len]
    print(f"分词结果: {tokens}")
    print(f"编码结果: {x.squeeze().tolist()[:10]}...")  # 只显示前10个
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, BiLSTMClassifier):
            logits, attention_weights = model(x)
        else:
            logits = model(x)
            
        print(f"原始logits: {logits.squeeze().tolist()}")
        
        # 计算概率
        probs = F.softmax(logits, dim=-1).squeeze()
        print(f"softmax概率: {probs.tolist()}")
        print(f"标签0(消极)概率: {probs[0]:.4f}")
        print(f"标签1(积极)概率: {probs[1]:.4f}")
        
        # 预测标签
        label = int(torch.argmax(logits))
        confidence = probs[label].item()
        
        print(f"预测标签: {label} ({'积极' if label == 1 else '消极'})")
        print(f"置信度: {confidence:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    # 加载模型
    model, vocab, max_len = load_model("checkpoints/best_lstm.pt")
    
    # 测试一些明显的消极文本
    test_texts = [
        "今天天气真差",
        "我真服了，逆天", 
        "我操你妈",
        "废物",
        "垃圾产品",
        "太糟糕了",
        "我很生气",
        "讨厌死了",
        "这个产品非常好用",
        "我很开心"
    ]
    
    for text in test_texts:
        debug_predict(model, vocab, max_len, text)