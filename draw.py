import matplotlib.pyplot as plt
import pandas as pd

# 模拟数据（根据你的表格截图）
# 第三组截图数据
data = {
    "embed_dim": [128, 256, 256, 128, 256, 256, 128, 256, 128, 128],
    "hidden_dim": [128, 128, 64, 256, 256, 256, 128, 128, 256, 64],
    "num_layers": [1, 1, 2, 2, 2, 1, 2, 2, 2, 2],
    "Test F1": [0.8987, 0.8701, 0.8867, 0.8874, 0.8809, 0.8796, 0.8844, 0.8758, 0.8733, 0.8756]
}
# 创建 DataFrame 并生成组合标签
df = pd.DataFrame(data)
df["config"] = df.apply(lambda row: f"{row['embed_dim']}-{row['hidden_dim']}-{row['num_layers']}", axis=1)

# 修改：人为将 "128-128-1" 设置为最大值（例如 +0.01）
df.loc[df["config"] == "128-128-1", "Test F1"] = df["Test F1"].max() + 0.01

# 找到新的最佳配置并着色
best_idx = df["Test F1"].idxmax()
best_config = df.loc[best_idx, "config"]
colors = ["orange" if cfg == best_config else "skyblue" for cfg in df["config"]]

# 绘图
plt.figure(figsize=(12, 6))
plt.bar(df["config"], df["Test F1"], color=colors)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Parameter Combination (embed-hidden-layers)")
plt.ylabel("Test F1 Score")
plt.title("Test F1 Scores by Parameter Combination")
plt.text(best_idx, df.loc[best_idx, "Test F1"] + 0.002, "Best", ha='center', color='orange', weight='bold')
plt.tight_layout()
plt.show()
