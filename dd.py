import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 构造嵌入维度与 F1 分数的数据（来自截图）
embedding_data = {
    "embedding_dim": [128, 256, 64],
    "f1_score": [0.874, 0.8625, 0.8713]
}

df_embed = pd.DataFrame(embedding_data)

# 找到最佳 F1 组合
best_idx = df_embed["f1_score"].idxmax()
best_config = df_embed.loc[best_idx, "embedding_dim"]
colors = ["orange" if dim == best_config else "skyblue" for dim in df_embed["embedding_dim"]]

# 绘图
plt.figure(figsize=(8, 5))
plt.bar(df_embed["embedding_dim"].astype(str), df_embed["f1_score"], color=colors)
plt.xlabel("Embedding Dimension")
plt.ylabel("F1 Score")
plt.title("Test F1 Score by Embedding Dimension")
plt.text(best_idx, df_embed.loc[best_idx, "f1_score"] + 0.002, "Best", ha='center', color='orange', weight='bold')
plt.tight_layout()
plt.show()
