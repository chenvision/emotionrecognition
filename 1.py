# Python 生成 tokenizer_dir 的做法
import os, shutil, json

os.makedirs("tokenizer_dir", exist_ok=True)
shutil.copy("spm_emotion.model", "tokenizer_dir/tokenizer.model")

with open("tokenizer_dir/tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump({"model_type": "bpe"}, f)

with open("tokenizer_dir/special_tokens_map.json", "w", encoding="utf-8") as f:
    json.dump({
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>"
    }, f)
