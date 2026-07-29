"""
模型测试脚本
功能：加载训练好的模型，在 CIFAR-10 测试集评估，输出准确率、混淆矩阵、分类报告
运行命令：python test.py
"""
import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from model import create_model

# 基础配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
# 训练保存的最优模型路径
MODEL_PATH = os.path.join(PROJECT_DIR, "cifar10_cnn_best.pth")
IMG_SIZE = 32
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试数据集
def load_test_data():
    # 测试集预处理和验证集一致，无增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    # CIFAR-10 官方测试集：10000 张
    test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"测试集总图片数: {len(test_dataset)} 识别类别: {test_dataset.classes}")
    return test_loader, test_dataset

# 完整遍历测试集，收集所有预测、真实标签、概率
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []    # 全部预测结果
    all_labels = []   # 全部真实标签
    all_probs = []    # 全部类别概率
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # softmax 把输出转为 0~1 概率
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        # 转 cpu numpy 保存
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# 绘制混淆矩阵图片（支持任意类别数量）
def plot_confusion_matrix(cm, class_names, save_path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(n), yticks=range(n),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='真实标签', xlabel='预测标签')
    # 旋转 x 轴标签避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # 在格子里写数字
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵图已保存: {save_path}")

# 随机抽取样本，可视化预测结果（绿色正确，红色错误）
def plot_prediction_samples(dataset, preds, labels, probs, class_names, save_path, n=12):
    # 分开正确、错误样本索引
    correct_idx = np.where(preds == labels)[0]
    wrong_idx = np.where(preds != labels)[0]
    random.seed(42)
    n_correct = min(n // 2, len(correct_idx))
    n_wrong = min(n - n_correct, len(wrong_idx))
    selected = random.sample(list(correct_idx), n_correct) + random.sample(list(wrong_idx), n_wrong)
    cols = 6
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    for i, idx in enumerate(selected):
        # 反标准化恢复原图色彩
        img = dataset[idx][0].numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)
        axes[i].imshow(img)
        correct = labels[idx] == preds[idx]
        color = 'green' if correct else 'red'
        axes[i].set_title(
            f"真实:{class_names[labels[idx]]}\n预测:{class_names[preds[idx]]} ({probs[idx][preds[idx]]:.2f})",
            fontsize=8, color=color
        )
        axes[i].axis('off')
    # 空白子图关闭坐标轴
    for i in range(len(selected), len(axes)):
        axes[i].axis('off')
    plt.suptitle('绿色=预测正确 红色=预测错误', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测样本可视化图已保存: {save_path}")

# 测试主函数
def test():
    print("=" * 50)
    print("CIFAR-10 模型测试集评估开始")
    print("=" * 50)
    test_loader, test_dataset = load_test_data()
    # 判断模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}，请先运行 train.py 训练！")
        sys.exit(1)
    # 加载模型（10 个类别）
    model = create_model(num_classes=len(test_dataset.classes), pretrained_path=MODEL_PATH).to(DEVICE)
    # 获取全部预测结果
    preds, labels, probs = evaluate(model, test_loader, DEVICE)
    # 整体准确率
    acc = (preds == labels).mean()
    print(f"\n测试集整体准确率: {acc:.4f} ({acc*100:.2f}%)")
    # 生成混淆矩阵
    cm = confusion_matrix(labels, preds)
    print(f"\n混淆矩阵（行=真实，列=预测）：")
    print(cm)
    # 单类别准确率
    print("\n各类别识别准确率：")
    for i, name in enumerate(test_dataset.classes):
        tp = cm[i, i]
        total = cm[i, :].sum()
        class_acc = tp / total * 100 if total > 0 else 0
        print(f"  {name:12s}: {class_acc:5.2f}% ({tp}/{total})")
    # 输出精确率、召回率、F1 分数
    print("\n分类评估报告：")
    print(classification_report(labels, preds, target_names=test_dataset.classes, digits=4))
    # 保存两张可视化图片
    plot_confusion_matrix(cm, test_dataset.classes, os.path.join(PROJECT_DIR, "confusion_matrix.png"))
    plot_prediction_samples(test_dataset, preds, labels, probs, test_dataset.classes, os.path.join(PROJECT_DIR, "prediction_samples.png"))

if __name__ == "__main__":
    test()
