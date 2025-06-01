"""情感分类模型训练脚本 - 支持LSTM、GRU、TextCNN三种模型"""
from __future__ import annotations  # 允许在注解中使用类型提示
import argparse  # 命令行参数解析器
import torch
from torch import nn, optim  # 神经网络和优化器
from torch.nn import functional as F
from tqdm import tqdm  # 进度条
from pathlib import Path  # 路径操作
from quick_test import quick_test, load_model

from data_loader import get_dataloaders  # 数据加载器
from utils import compute_metrics, DEVICE
from model_lstm import BiLSTMClassifier
from model_gru import BiGRUClassifier
from model_textcnn import TextCNNClassifier

# ======================================================================
# Focal Loss 实现
# ======================================================================

class FocalLoss(nn.Module):
    """Focal Loss实现，用于处理类别不平衡和难分样本
    
    Args:
        alpha: 类别权重，可以是float或tensor
        gamma: 聚焦参数，gamma越大越关注难分样本
        reduction: 损失聚合方式
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 计算focal权重
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用alpha权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ======================================================================
# 训练和评估函数
# ======================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """训练一个epoch
    
    Args:
        model: 神经网络模型
        loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
    
    Returns:
        float: 平均训练损失
    """
    model.train()  # 设置为训练模式
    total_loss = 0
    
    for x, y in loader:
        # 将数据移动到GPU/CPU
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播（忽略注意力权重）
        logits, _ = model(x)
        
        # 计算损失
        loss = criterion(logits, y)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item() * y.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """评估模型性能
    
    Args:
        model: 神经网络模型
        loader: 验证/测试数据加载器
        criterion: 损失函数
    
    Returns:
        dict: 包含损失、准确率、F1分数等指标的字典
    """
    model.eval()  # 设置为评估模式
    total_loss = 0
    preds, labels = [], []
    attention_weights_list = []
    
    for x, y in loader:
        # 将数据移动到GPU/CPU
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # 前向传播
        logits, attention_weights = model(x)
        
        # 计算损失
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        
        # 收集预测结果和真实标签
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        
        # 保存注意力权重用于可视化
        attention_weights_list.append(attention_weights.cpu())
    
    # 计算各种评估指标
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["attention_weights"] = torch.cat(attention_weights_list, dim=0)
    
    return metrics


# ======================================================================
# 主训练函数
# ======================================================================

def main():
    """主训练函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="情感分类模型训练")
    
    # 数据相关参数
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="训练数据目录")
    parser.add_argument("--max_len", type=int, default=128, 
                       help="文本最大长度")
    
    # 模型相关参数
    parser.add_argument("--model", choices=["lstm", "gru", "textcnn"], 
                       default="lstm", help="模型类型")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=30, 
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-4, 
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                       help="L2正则化系数")
    parser.add_argument("--patience", type=int, default=7, 
                       help="早停耐心值（连续多少轮不改善就停止）")
    parser.add_argument("--min_improvement", type=float, default=1e-4, 
                       help="F1分数改善的最小阈值")
    parser.add_argument("--resume_best", action="store_true", 
                       help="自动基于已有最佳模型的F1分数作为基准继续训练")
    parser.add_argument("--overwrite", action="store_true", 
                       help="强制覆盖已有模型，从0开始计算F1基准")
    
    # 损失函数相关参数
    parser.add_argument("--loss_type", choices=["crossentropy", "focal"], 
                       default="crossentropy", help="损失函数类型")
    parser.add_argument("--focal_gamma", type=float, default=2.0, 
                       help="Focal Loss的gamma参数，控制难分样本的关注度")
    parser.add_argument("--negative_weight_boost", type=float, default=1.5, 
                       help="消极类别权重增强倍数，用于进一步缓解类别不平衡")
    
    # 保存相关参数
    parser.add_argument("--save_path", type=str, default="checkpoints", 
                       help="模型保存路径")
    
    args = parser.parse_args()

    # ======================================================================
    # 数据加载
    # ======================================================================
    print("正在加载数据...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        args.data_dir, args.batch_size, args.max_len
    )
    
    # 打印数据集信息
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"词汇表大小: {len(vocab)}")
    print(f"最大序列长度: {args.max_len}")
    
    # 创建模型保存目录
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 模型初始化
    # ======================================================================
    print(f"正在初始化{args.model.upper()}模型...")
    
    # 根据命令行参数选择模型类
    if args.model == "lstm":
        model_cls = BiLSTMClassifier
    elif args.model == "gru":
        model_cls = BiGRUClassifier
    else:
        model_cls = TextCNNClassifier
    
    # 创建模型实例并移动到设备
    model = model_cls(len(vocab)).to(DEVICE)
    print(f"模型已创建，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ======================================================================
    # 损失函数和优化器设置
    # ======================================================================
    print("正在设置训练组件...")
    
    # 计算类别权重以解决数据不平衡问题
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    print("正在计算类别权重...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    # 使用sklearn计算平衡权重
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    # 增强消极类别权重（给消极类别更高的权重）
    negative_boost = getattr(args, 'negative_weight_boost', 1.5)  # 默认1.5倍
    class_weights[0] *= negative_boost  # 消极类别权重增强
    
    print(f"原始类别权重计算完成")
    print(f"增强后类别权重 - 消极: {class_weights[0]:.4f}, 积极: {class_weights[1]:.4f}")
    print(f"消极类别权重增强倍数: {negative_boost}")
    
    # 根据参数选择损失函数
    loss_type = getattr(args, 'loss_type', 'crossentropy')  # 默认交叉熵
    
    if loss_type == 'focal':
        # 使用Focal Loss
        focal_gamma = getattr(args, 'focal_gamma', 2.0)  # 默认gamma=2.0
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        print(f"使用Focal Loss - gamma: {focal_gamma}, alpha权重已应用")
    else:
        # 使用带权重的交叉熵损失
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权交叉熵损失")
    
    # 初始化优化器（Adam + L2正则化）
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 初始化学习率调度器（当F1不提升时降低学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # 监控指标越大越好
        factor=0.5,      # 学习率衰减因子
        patience=2,      # 等待轮数
        verbose=True     # 打印调整信息
    )
    
    # ======================================================================
    # 检查已有模型并设置基准
    # ======================================================================
    best_f1 = 0.0
    best_epoch = 0
    no_improve_count = 0
    improvement_threshold = args.min_improvement  # F1改善的最小阈值
    
    # 检查是否存在已有的最佳模型
    existing_model_path = Path(args.save_path) / f"best_{args.model}.pt"
    if existing_model_path.exists():
        try:
            print(f"发现已有模型: {existing_model_path}")
            existing_ckpt = torch.load(existing_model_path, map_location=DEVICE)
            existing_f1 = existing_ckpt.get("best_f1", 0.0)
            existing_epoch = existing_ckpt.get("epoch", 0)
            
            print(f"已有模型最佳F1: {existing_f1:.6f} (第{existing_epoch}轮)")
            
            # 根据命令行参数决定是否继续训练
            if args.overwrite:
                print("✓ 使用 --overwrite 参数，将从0开始计算，新训练的模型将覆盖已有模型")
            elif args.resume_best:
                best_f1 = existing_f1
                best_epoch = existing_epoch
                print(f"✓ 使用 --resume_best 参数，将使用已有模型F1分数 {best_f1:.6f} 作为基准")
                print("  只有超过此分数的模型才会被保存")
            else:
                # 交互式询问
                response = input("是否基于已有模型的F1分数作为基准继续训练？(y/n，默认n): ").strip().lower()
                if response in ['y', 'yes', '是']:
                    best_f1 = existing_f1
                    best_epoch = existing_epoch
                    print(f"✓ 将使用已有模型F1分数 {best_f1:.6f} 作为基准")
                    print("  只有超过此分数的模型才会被保存")
                else:
                    print("✓ 将从0开始计算，新训练的模型将覆盖已有模型")
        except Exception as e:
            print(f"⚠️  读取已有模型失败: {e}")
            print("将从0开始训练")
    else:
        print("未发现已有模型，将从头开始训练")
    
    print(f"\n训练设置完成 - 学习率: {args.lr}, 权重衰减: {args.weight_decay}")
    print(f"早停设置 - 耐心值: {args.patience}, F1改善阈值: {improvement_threshold}")
    print(f"当前基准F1: {best_f1:.6f}")
    if args.patience == 0:
        print("⚠️  早停已禁用 (patience=0)")
    print("="*60)
    
    # ======================================================================
    # 训练循环
    # ======================================================================
    print("开始训练...")
    
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # 在验证集上评估
        val_metrics = evaluate(model, val_loader, criterion)
        
        # 打印训练进度
        print(
            f"Epoch {epoch:2d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # 学习率调度（基于验证F1分数）
        scheduler.step(val_metrics['f1'])
        
        # 检查是否有显著改善（使用阈值避免浮点数精度问题）
        current_f1 = val_metrics["f1"]
        f1_improvement = current_f1 - best_f1
        
        print(f"  当前F1: {current_f1:.6f}, 最佳F1: {best_f1:.6f}, 改善: {f1_improvement:.6f}")
        
        if f1_improvement > improvement_threshold:
            print(f"  ✓ 验证F1显著提升: {best_f1:.6f} -> {current_f1:.6f} (改善 {f1_improvement:.6f})")
            best_f1 = current_f1
            best_epoch = epoch
            no_improve_count = 0
            
            # 保存最佳模型
            val_attention = val_metrics.pop("attention_weights")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "args": vars(args),
                    "attention_weights": val_attention,
                    "best_f1": best_f1,
                    "epoch": epoch,
                },
                Path(args.save_path) / f"best_{args.model}.pt",
            )
            print(f"  ✓ 模型已保存到 {args.save_path}/best_{args.model}.pt")
        else:
            no_improve_count += 1
            if f1_improvement <= 0:
                print(f"  ✗ 验证F1下降或无变化 (改善: {f1_improvement:.6f})")
            else:
                print(f"  ✗ 验证F1提升不显著 (改善: {f1_improvement:.6f} <= 阈值: {improvement_threshold})")
            
            print(f"  早停计数: {no_improve_count}/{args.patience}")
            
            # 早停检查（如果patience为0则禁用早停）
            if args.patience > 0 and no_improve_count >= args.patience:
                print(f"\n🛑 早停触发！连续{args.patience}轮验证F1未显著提升，训练结束。")
                print(f"   最佳F1分数: {best_f1:.6f} (第{best_epoch}轮)")
                print(f"   当前F1分数: {current_f1:.6f} (第{epoch}轮)")
                print(f"   总训练轮数: {epoch}/{args.epochs}")
                break
        
        print("-" * 60)

    # ======================================================================
    # 最终测试评估
    # ======================================================================
    print("\n正在加载最佳模型进行测试...")
    
    # 加载最佳模型
    ckpt_path = Path(args.save_path) / f"best_{args.model}.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    
    # 在测试集上评估
    test_metrics = evaluate(model, test_loader, criterion)
    
    print("="*60)
    print("最终测试结果:")
    print(f"  测试损失: {test_metrics['loss']:.4f}")
    print(f"  测试准确率: {test_metrics['accuracy']:.4f}")
    print(f"  测试F1分数: {test_metrics['f1']:.4f}")
    print(f"  测试精确率: {test_metrics['precision']:.4f}")
    print(f"  测试召回率: {test_metrics['recall']:.4f}")
    print(f"\n最佳模型保存在: {ckpt_path}")
    print(f"训练完成于第{ckpt['epoch']}轮，最佳验证F1: {ckpt['best_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
