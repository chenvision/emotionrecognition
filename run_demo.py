"""Quick interactive demo."""
import argparse
from predict import load_model, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    model, vocab, max_len = load_model(args.ckpt)
    print("请输入一句话 (按 Ctrl+C 退出):")
    try:
        while True:
            txt = input(">>> ").strip()
            if not txt:
                continue
            label, conf, weights, tokens = predict(model, vocab, max_len, txt)
            label_str = "积极" if label == 1 else "消极"
            print(f"  -> {label_str} ({conf:.3f})")
    except KeyboardInterrupt:
        print("\n退出")


if __name__ == "__main__":
    main()
