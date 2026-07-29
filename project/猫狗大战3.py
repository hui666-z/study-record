"""
猫狗分类测试脚本
功能：加载训练好的模型，在独立测试集评估，输出准确率、混淆矩阵、精确率/召回率/F1，生成可视化图片
"""
import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 分类评估指标工具
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from model import create_model

# 全局配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
TEST_DIR = os.path.join(DATA_ROOT, "test_set", "test_set")
MODEL_PATH = os.path.join(PROJECT_DIR, "cat_dog_cnn_best.pth")
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试集（预处理和验证集完全一致，无增强）
def load_test_data():
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"测试集总图片: {len(test_dataset)} 张，类别: {test_dataset.classes}")
    return test_loader, test_dataset

# 遍历测试集，收集全部预测、真实标签、概率
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1) # logits转为0~1概率
        preds = outputs.argmax(dim=1)
        # 转cpu numpy存入列表
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# 绘制混淆矩阵热力图
def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # 在格子里写入数字
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20, color=text_color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"混淆矩阵图片已保存: {save_path}")

# 随机抽取样本可视化预测结果（绿色正确、红色错误）
def plot_prediction_samples(dataset, preds, labels, probs, class_names, save_path, n=12):
    # 分开正确、错误样本索引
    correct_idx = np.where(preds == labels)[0]
    wrong_idx = np.where(preds != labels)[0]
    random.seed(42)
    # 各取一半
    n_correct = min(n//2, len(correct_idx))
    n_wrong = min(n - n_correct, len(wrong_idx))
    selected = random.sample(list(correct_idx), n_correct) + random.sample(list(wrong_idx), n_wrong)

    cols = 6
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*3))
    axes = axes.flatten() if rows*cols > 1 else [axes]

    # 反归一化恢复图片原始色彩
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i, idx in enumerate(selected):
        img_tensor = dataset[idx][0].numpy().transpose(1, 2, 0)
        img = np.clip(img_tensor * std + mean, 0, 1)
        axes[i].imshow(img)
        is_right = labels[idx] == preds[idx]
        title_color = 'green' if is_right else 'red'
        axes[i].set_title(f"True:{class_names[labels[idx]]}\nPred:{class_names[preds[idx]]}({probs[idx][preds[idx]]:.2f})", fontsize=8, color=title_color)
        axes[i].axis('off')
    # 空白子图关闭坐标轴
    for i in range(len(selected), len(axes)):
        axes[i].axis('off')
    plt.suptitle("Green=Correct | Red=Wrong", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"预测样本可视化图片已保存: {save_path}")

# 测试主函数
def test():
    print("="*50)
    print("开始测试集模型评估")
    print("="*50)
    test_loader, test_dataset = load_test_data()
    # 判断模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}，请先运行train.py训练！")
        sys.exit(1)
    # 加载模型
    model = create_model(pretrained_path=MODEL_PATH).to(DEVICE)
    # 预测全部测试集
    preds, labels, probs = evaluate(model, test_loader, DEVICE)
    # 整体准确率
    total_acc = (preds == labels).mean()
    print(f"\n测试集整体准确率: {total_acc:.4f} ({total_acc*100:.2f}%)")

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    print("\n混淆矩阵：")
    print(f"真实猫：预测猫{cm[0,0]} 预测狗{cm[0,1]}")
    print(f"真实狗：预测猫{cm[1,0]} 预测狗{cm[1,1]}")

    # 单类别准确率
    cat_acc = cm[0,0]/(cm[0,0]+cm[0,1]) * 100
    dog_acc = cm[1,1]/(cm[1,0]+cm[1,1]) * 100
    print(f"\n猫识别准确率: {cat_acc:.2f}%")
    print(f"狗识别准确率: {dog_acc:.2f}%")

    # 精确率、召回率、F1分数
    print("\n分类指标报告：")
    print(classification_report(labels, preds, target_names=test_dataset.classes, digits=4))

    # 保存两张可视化图
    plot_confusion_matrix(cm, test_dataset.classes, os.path.join(PROJECT_DIR, "confusion_matrix.png"))
    plot_prediction_samples(test_dataset, preds, labels, probs, test_dataset.classes, os.path.join(PROJECT_DIR, "prediction_samples.png"))

if __name__ == "__main__":
    test()