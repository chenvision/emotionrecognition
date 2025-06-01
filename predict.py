"""Load checkpoint and predict single sentence(s)."""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
from utils import tokenize, DEVICE
from model_lstm import BiLSTMClassifier
from model_gru import BiGRUClassifier
from model_textcnn import TextCNNClassifier


def load_model(ckpt_path: str):
    # 先加载模型信息
    ckpt_info = torch.load(ckpt_path, map_location=DEVICE)
    vocab = ckpt_info["vocab"]
    args = ckpt_info["args"]
    
    # 创建模型实例
    if args["model"] == "lstm":
        model_cls = BiLSTMClassifier
    elif args["model"] == "gru":
        model_cls = BiGRUClassifier
    else:
        model_cls = TextCNNClassifier
    model = model_cls(len(vocab)).to(DEVICE)
    
    # 加载模型权重
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, vocab, args["max_len"]


def encode(text: str, vocab: dict[str, int], max_len: int):
    if not text or not text.strip():
        raise ValueError("输入文本不能为空")
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("分词结果为空，请检查输入文本")
    ids = [vocab.get(t, 1) for t in tokens][:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def predict(model, vocab, max_len: int, text: str):
    x = encode(text, vocab, max_len).to(DEVICE)
    tokens = tokenize(text)[:max_len]
    
    # 区分LSTM/GRU和TextCNN模型的输出处理
    if isinstance(model, (BiLSTMClassifier, BiGRUClassifier)):
        logits, attention_weights = model(x)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
        label = int(torch.argmax(logits))
        
        # 获取注意力权重并可视化
        weights = attention_weights.squeeze().cpu().tolist()
        weights = weights[:len(tokens)]  # 只取实际token的权重
        
        # 创建热力图
        plt.figure(figsize=(12, 3))
        sns.heatmap([weights], cmap='YlOrRd', xticklabels=tokens, yticklabels=False)
        plt.title('注意力权重可视化')
        plt.xticks(rotation=0, fontsize=10)  # 横向显示文本，增大字号
        plt.tight_layout()
        
        return label, probs[label], weights, tokens
    else:
        logits = model(x)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
        label = int(torch.argmax(logits))
        return label, probs[label], None, tokens


def main():
    parser = argparse.ArgumentParser(description="Predict emotion of a sentence")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=str, help="Path to save attention visualization", default="attention.png")
    args = parser.parse_args()
    
    try:
        if not args.text.strip():
            raise ValueError("请提供非空的输入文本")
            
        model, vocab, max_len = load_model(args.ckpt)
        label, conf, weights, tokens = predict(model, vocab, max_len, args.text)
        label_str = "积极" if label == 1 else "消极"
        
        print(f"输入文本: {args.text}")
        print(f"情绪类型: {label_str} | 置信度: {conf:.3f}")
        
        # 只有LSTM模型才显示注意力信息
        if isinstance(model, BiLSTMClassifier) and weights is not None:
            print("\n注意力分布:")
            for token, weight in zip(tokens, weights):
                print(f"{token}: {weight:.3f}")
            plt.savefig(args.output)
            print(f"\n注意力可视化已保存至: {args.output}")
            
    except FileNotFoundError as e:
        print(f"错误：找不到模型文件 {args.ckpt}")
    except ValueError as e:
        print(f"错误：{str(e)}")
    except Exception as e:
        print(f"发生错误：{str(e)}")



if __name__ == "__main__":
    main()