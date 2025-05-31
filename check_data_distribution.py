import pandas as pd

# 读取训练数据
df = pd.read_csv('data/train.tsv', sep='\t', header=None, names=['text', 'label'])

print('训练数据标签分布:')
print(df['label'].value_counts())
print(f'总样本数: {df.shape[0]}')
print(f'积极样本(标签1)数量: {df[df["label"]==1].shape[0]}')
print(f'消极样本(标签0)数量: {df[df["label"]==0].shape[0]}')
print(f'积极样本比例: {df[df["label"]==1].shape[0]/df.shape[0]:.3f}')
print(f'消极样本比例: {df[df["label"]==0].shape[0]/df.shape[0]:.3f}')

# 检查一些消极样本
print('\n一些消极样本示例:')
negative_samples = df[df['label']==0]['text'].head(10)
for i, text in enumerate(negative_samples, 1):
    print(f'{i}. {text}')