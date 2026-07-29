"""
猫狗识别训练脚本
功能：加载数据集、划分训练/验证集、数据增强、训练循环、保存最优模型、绘制loss/acc曲线
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# 导入我们自己写的模型文件
from model import create_model

# ===================== 全局超参数配置区（初学者直接改这里） =====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前项目根目录
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_ROOT, "training_set", "training_set")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "cat_dog_cnn_best.pth")  # 最优模型保存路径

IMG_SIZE = 224       # 统一图片尺寸
BATCH_SIZE = 32      # 一次读32张图片训练
EPOCHS = 30          # 完整遍历数据集30轮
LEARNING_RATE = 0.001# 学习率，控制参数更新幅度
WEIGHT_DECAY = 1e-4  # L2正则，防止过拟合
VAL_RATIO = 0.15     # 15%数据作为验证集
RANDOM_SEED = 42     # 随机种子，保证每次运行结果一致

# 自动优先使用GPU，无GPU则用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定所有随机种子，实验可复现
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 数据预处理：训练集带增强，验证集仅标准化
def get_data_transforms():
    # 训练集增强：随机裁剪、翻转、旋转、色彩抖动，扩充数据防过拟合
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), # 随机裁剪70%-100%区域
        transforms.RandomHorizontalFlip(p=0.5), # 50%概率水平翻转
        transforms.RandomRotation(degrees=15),  # ±15度随机旋转
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), # 亮度对比度随机变化
        transforms.ToTensor(), # 图片转张量 HWC→CHW，像素0~255转0~1
        # ImageNet全局均值方差标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 验证集：不做任何增强，保证评估公平
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform

# 加载数据集、划分训练/验证集、生成数据加载器
def prepare_data():
    train_transform, eval_transform = get_data_transforms()
    # 先不带transform读取全部图片，仅用来划分索引
    full_dataset = datasets.ImageFolder(root=TRAIN_DIR)
    print(f"识别类别: {full_dataset.classes}")
    print(f"训练集总图片数量: {len(full_dataset)}")

    # 按比例划分训练、验证集索引
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # 给训练、验证索引分别绑定不同预处理
    train_dataset = Subset(datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform), train_subset.indices)
    val_dataset = Subset(datasets.ImageFolder(root=TRAIN_DIR, transform=eval_transform), val_subset.indices)
    print(f"训练图片: {len(train_dataset)} 张 | 验证图片: {len(val_dataset)} 张")

    # DataLoader：批量读取数据
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, full_dataset.classes

# 单轮训练函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # 开启dropout、BN训练模式
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        # 数据迁移到GPU/CPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 标准训练五步流程
        optimizer.zero_grad()       # 1.清空上一轮梯度
        outputs = model(inputs)     # 2.前向传播得到预测分数
        loss = criterion(outputs, labels) # 3.计算损失
        loss.backward()             # 4.反向传播求梯度
        optimizer.step()            # 5.更新网络权重

        # 统计loss和准确率
        preds = outputs.argmax(dim=1) # 取分数最大的类别作为预测结果
        total_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total_samples += inputs.size(0)

        # 每50个batch打印一次中间信息
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_samples
            avg_acc = correct / total_samples
            print(f"  Batch[{batch_idx+1}/{len(loader)}] Loss:{avg_loss:.4f} Acc:{avg_acc:.4f}")

    epoch_loss = total_loss / total_samples
    epoch_acc = correct / total_samples
    return epoch_loss, epoch_acc

# 验证集评估函数（不更新权重）
@torch.no_grad() # 关闭梯度计算，节省显存、提速
def validate(model, loader, criterion, device):
    model.eval() # 关闭dropout，使用BN全局均值方差
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total_samples += inputs.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct / total_samples
    return epoch_loss, epoch_acc

# 绘制训练loss、准确率曲线并保存图片
def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss Curve')
    ax1.legend(); ax1.grid(alpha=0.3)
    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy Curve')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n训练曲线图片已保存至: {save_path}")

# 主训练入口
def train():
    print("="*50)
    print(f"训练设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    set_seed(RANDOM_SEED)

    # 1.加载数据
    train_loader, val_loader, class_names = prepare_data()
    # 2.初始化模型、损失函数、优化器、学习率调度器
    model = create_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # 二分类损失，自带softmax
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 验证loss连续4轮不下降，学习率减半，最低1e-6
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    # 记录每轮训练数据
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0 # 记录最优验证准确率
    total_start = time.time()

    # 循环训练每一轮
    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n===== Epoch [{epoch}/{EPOCHS}] 当前学习率: {current_lr:.6f} =====")
        epoch_start = time.time()

        # 训练一轮 + 验证一轮
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # 根据验证loss调整学习率
        scheduler.step(val_loss)

        # 保存本轮数据
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印本轮结果
        print(f"训练 Loss: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"验证 Loss: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"本轮耗时: {time.time() - epoch_start:.1f} s")

        # 保存验证集效果最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 保存最优模型，当前最佳验证准确率: {best_acc:.4f}")

    # 全部轮次结束
    total_min = (time.time() - total_start) / 60
    print(f"\n训练全部完成！总耗时 {total_min:.1f} 分钟，最高验证准确率: {best_acc:.4f}")
    # 绘制曲线
    plot_training_history(history, os.path.join(PROJECT_DIR, "training_history.png"))

# 程序入口
if __name__ == "__main__":
    train()