#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量检查和修复脚本

功能:
1. 检查数据标签一致性
2. 识别可能的标签错误
3. 分析数据分布
4. 生成数据质量报告
5. 提供标签修复建议
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.positive_keywords = {
            '强正面': ['太棒了', '非常好', '很棒', '优秀', '完美', '杰出', '卓越', '惊艳'],
            '正面': ['好', '不错', '可以', '还行', '满意', '喜欢', '开心', '高兴', '快乐', '愉快']
        }
        self.negative_keywords = {
            '强负面': ['垃圾', '废物', '糟糕', '恶心', '讨厌', '愤怒', '气愤', '恼火', '操'],
            '负面': ['坏', '差', '不好', '不行', '失望', '生气', '难过', '郁闷', '烦']
        }
        
    def load_data(self):
        """加载所有数据文件"""
        data = {}
        for split in ['train', 'dev', 'test']:
            file_path = self.data_dir / f"{split}.tsv"
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
                data[split] = df
                print(f"加载 {split}.tsv: {len(df)} 条样本")
            else:
                print(f"警告: {file_path} 不存在")
        return data
    
    def analyze_basic_stats(self, data):
        """分析基本统计信息"""
        print("\n" + "="*60)
        print("基本统计信息")
        print("="*60)
        
        for split, df in data.items():
            print(f"\n{split.upper()} 集:")
            print(f"  总样本数: {len(df)}")
            print(f"  标签分布:")
            label_counts = df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                percentage = count / len(df) * 100
                label_name = "消极" if label == 0 else "积极"
                print(f"    {label} ({label_name}): {count} ({percentage:.1f}%)")
            
            print(f"  文本长度统计:")
            text_lengths = df['text'].str.len()
            print(f"    平均长度: {text_lengths.mean():.1f}")
            print(f"    最短: {text_lengths.min()}")
            print(f"    最长: {text_lengths.max()}")
            print(f"    中位数: {text_lengths.median():.1f}")
    
    def check_keyword_consistency(self, data):
        """检查关键词与标签的一致性"""
        print("\n" + "="*60)
        print("关键词-标签一致性检查")
        print("="*60)
        
        inconsistencies = []
        
        for split, df in data.items():
            print(f"\n{split.upper()} 集不一致样本:")
            split_inconsistencies = []
            
            for idx, row in df.iterrows():
                text = row['text']
                label = row['label']
                
                # 检查正面关键词但标签为负面
                for category, keywords in self.positive_keywords.items():
                    for keyword in keywords:
                        if keyword in text and label == 0:  # 正面词但负面标签
                            issue = {
                                'split': split,
                                'index': idx,
                                'text': text,
                                'label': label,
                                'expected_label': 1,
                                'keyword': keyword,
                                'category': category,
                                'issue_type': 'positive_keyword_negative_label'
                            }
                            split_inconsistencies.append(issue)
                            print(f"  [{idx}] 正面词'{keyword}'但标签为消极: {text}")
                
                # 检查负面关键词但标签为正面
                for category, keywords in self.negative_keywords.items():
                    for keyword in keywords:
                        if keyword in text and label == 1:  # 负面词但正面标签
                            issue = {
                                'split': split,
                                'index': idx,
                                'text': text,
                                'label': label,
                                'expected_label': 0,
                                'keyword': keyword,
                                'category': category,
                                'issue_type': 'negative_keyword_positive_label'
                            }
                            split_inconsistencies.append(issue)
                            print(f"  [{idx}] 负面词'{keyword}'但标签为积极: {text}")
            
            inconsistencies.extend(split_inconsistencies)
            if not split_inconsistencies:
                print("  未发现明显不一致")
        
        return inconsistencies
    
    def analyze_text_patterns(self, data):
        """分析文本模式"""
        print("\n" + "="*60)
        print("文本模式分析")
        print("="*60)
        
        # 合并所有数据进行分析
        all_texts = []
        all_labels = []
        for df in data.values():
            all_texts.extend(df['text'].tolist())
            all_labels.extend(df['label'].tolist())
        
        # 分析每个类别的高频词
        positive_texts = [text for text, label in zip(all_texts, all_labels) if label == 1]
        negative_texts = [text for text, label in zip(all_texts, all_labels) if label == 0]
        
        print(f"\n积极文本高频词 (共{len(positive_texts)}条):")
        pos_words = self._get_frequent_words(positive_texts, top_k=20)
        for word, count in pos_words:
            print(f"  {word}: {count}")
        
        print(f"\n消极文本高频词 (共{len(negative_texts)}条):")
        neg_words = self._get_frequent_words(negative_texts, top_k=20)
        for word, count in neg_words:
            print(f"  {word}: {count}")
        
        return pos_words, neg_words
    
    def _get_frequent_words(self, texts, top_k=20):
        """获取高频词"""
        word_counter = Counter()
        for text in texts:
            # 简单分词（可以替换为更好的分词器）
            words = list(jieba.cut(text))
            # 过滤停用词和标点
            words = [w for w in words if len(w) > 1 and w.isalnum()]
            word_counter.update(words)
        return word_counter.most_common(top_k)
    
    def cluster_analysis(self, data, n_clusters=4):
        """聚类分析找出异常样本"""
        print(f"\n" + "="*60)
        print(f"聚类分析 (k={n_clusters})")
        print("="*60)
        
        # 合并训练数据进行聚类
        if 'train' not in data:
            print("警告: 没有训练数据，跳过聚类分析")
            return
        
        df = data['train']
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        X = vectorizer.fit_transform(texts)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"轮廓系数: {silhouette_avg:.3f}")
        
        # 分析每个聚类的标签分布
        cluster_analysis = defaultdict(lambda: {'positive': 0, 'negative': 0, 'samples': []})
        
        for i, (text, label, cluster) in enumerate(zip(texts, labels, cluster_labels)):
            if label == 1:
                cluster_analysis[cluster]['positive'] += 1
            else:
                cluster_analysis[cluster]['negative'] += 1
            
            cluster_analysis[cluster]['samples'].append((i, text, label))
        
        # 找出可能的异常聚类
        anomalies = []
        for cluster_id, info in cluster_analysis.items():
            total = info['positive'] + info['negative']
            pos_ratio = info['positive'] / total
            
            print(f"\n聚类 {cluster_id}: {total} 样本")
            print(f"  积极: {info['positive']} ({pos_ratio:.1%})")
            print(f"  消极: {info['negative']} ({1-pos_ratio:.1%})")
            
            # 如果聚类中某个标签占比很小，可能是异常
            if 0.1 < pos_ratio < 0.9:  # 混合聚类
                print(f"  ⚠️  混合聚类，可能包含标签错误")
                # 显示一些样本
                for idx, text, label in info['samples'][:3]:
                    label_name = "积极" if label == 1 else "消极"
                    print(f"    [{idx}] {label_name}: {text}")
                
                anomalies.extend([(idx, text, label) for idx, text, label in info['samples']])
        
        return anomalies
    
    def generate_report(self, data, inconsistencies, anomalies=None):
        """生成数据质量报告"""
        print("\n" + "="*60)
        print("数据质量报告")
        print("="*60)
        
        # 总体统计
        total_samples = sum(len(df) for df in data.values())
        total_inconsistencies = len(inconsistencies)
        
        print(f"\n总体概况:")
        print(f"  总样本数: {total_samples:,}")
        print(f"  发现的不一致样本: {total_inconsistencies}")
        print(f"  不一致比例: {total_inconsistencies/total_samples*100:.2f}%")
        
        # 按问题类型分组
        issue_types = defaultdict(int)
        for issue in inconsistencies:
            issue_types[issue['issue_type']] += 1
        
        print(f"\n问题类型分布:")
        for issue_type, count in issue_types.items():
            if issue_type == 'positive_keyword_negative_label':
                print(f"  正面词但负面标签: {count}")
            elif issue_type == 'negative_keyword_positive_label':
                print(f"  负面词但正面标签: {count}")
        
        # 建议
        print(f"\n修复建议:")
        if total_inconsistencies > 0:
            print(f"  1. 手动检查并修正 {total_inconsistencies} 个不一致样本")
            print(f"  2. 考虑扩充训练数据以平衡类别")
            print(f"  3. 使用更严格的数据标注指南")
            print(f"  4. 考虑使用半监督学习方法")
        else:
            print(f"  数据质量良好，未发现明显问题")
        
        if anomalies:
            print(f"  5. 检查聚类分析发现的 {len(anomalies)} 个可能异常样本")
    
    def save_inconsistencies(self, inconsistencies, output_path="data_issues.csv"):
        """保存不一致样本到文件"""
        if not inconsistencies:
            print("\n没有发现不一致样本，无需保存")
            return
        
        df = pd.DataFrame(inconsistencies)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n不一致样本已保存到: {output_path}")
        print(f"可以手动检查并修正这些样本")
    
    def plot_data_distribution(self, data, save_path="data_distribution.png"):
        """绘制数据分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 标签分布
        for i, (split, df) in enumerate(data.items()):
            if i >= 3:  # 最多显示3个数据集
                break
            
            row, col = i // 2, i % 2
            label_counts = df['label'].value_counts().sort_index()
            
            axes[row, col].bar(['消极 (0)', '积极 (1)'], label_counts.values, 
                              color=['red', 'blue'], alpha=0.7)
            axes[row, col].set_title(f'{split.upper()} 集标签分布')
            axes[row, col].set_ylabel('样本数')
            
            # 添加数值标签
            for j, v in enumerate(label_counts.values):
                axes[row, col].text(j, v + 0.01 * max(label_counts.values), 
                                   str(v), ha='center', va='bottom')
        
        # 文本长度分布
        if len(data) > 0:
            all_lengths = []
            for df in data.values():
                all_lengths.extend(df['text'].str.len().tolist())
            
            axes[1, 1].hist(all_lengths, bins=30, alpha=0.7, color='green')
            axes[1, 1].set_title('文本长度分布')
            axes[1, 1].set_xlabel('字符数')
            axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n数据分布图已保存到: {save_path}")
    
    def run_full_check(self):
        """运行完整的数据质量检查"""
        print("开始数据质量检查...")
        
        # 加载数据
        data = self.load_data()
        if not data:
            print("错误: 没有找到数据文件")
            return
        
        # 基本统计
        self.analyze_basic_stats(data)
        
        # 关键词一致性检查
        inconsistencies = self.check_keyword_consistency(data)
        
        # 文本模式分析
        self.analyze_text_patterns(data)
        
        # 聚类分析（仅对训练数据）
        anomalies = self.cluster_analysis(data)
        
        # 生成报告
        self.generate_report(data, inconsistencies, anomalies)
        
        # 保存结果
        self.save_inconsistencies(inconsistencies)
        self.plot_data_distribution(data)
        
        return data, inconsistencies, anomalies


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="数据质量检查")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建检查器
    checker = DataQualityChecker(args.data_dir)
    
    # 运行检查
    data, inconsistencies, anomalies = checker.run_full_check()
    
    print(f"\n数据质量检查完成！")
    print(f"检查了 {sum(len(df) for df in data.values())} 个样本")
    print(f"发现 {len(inconsistencies)} 个潜在问题")
    
    if inconsistencies:
        print("\n建议下一步操作:")
        print("1. 检查 data_issues.csv 文件中的问题样本")
        print("2. 手动修正标签错误")
        print("3. 使用修正后的数据重新训练模型")


if __name__ == "__main__":
    main()