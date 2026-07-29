"""
模型训练脚本
功能：加载 CIFAR-10 数据集、划分训练验证集、循环训练、保存最优模型、绘制训练曲线
运行命令：python train.py
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

# ====================== 全局配置超参数（初学者直接改这里） ======================
# 获取当前脚本所在文件夹绝对路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集根目录（CIFAR-10 数据在 data/cifar-10-batches-py/ 下）
DATA_ROOT = os.path.join(PROJECT_DIR, "data")
# 训练完成后最优模型保存路径
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "cifar10_cnn_best.pth")
IMG_SIZE = 32           # CIFAR-10 原始图片尺寸 32×32
BATCH_SIZE = 128        # CIFAR-10 可以适当增大批次
EPOCHS = 30             # 完整遍历数据集 30 轮
LEARNING_RATE = 0.001   # 学习率，控制参数更新幅度
WEIGHT_DECAY = 1e-4     # L2 正则化，防止过拟合
VAL_RATIO = 0.15        # 拿 15% 训练集当作验证集（CIFAR-10 共 5 万张，即 7500 张验证）
RANDOM_SEED = 42        # 随机种子，固定后每次划分数据一致，实验可复现
# 自动判断设备：有 GPU 用 cuda，没有用 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定所有随机种子，保证每次运行结果相同
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果有 GPU，同步固定 GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 数据预处理函数：训练集增强，验证集仅标准化
def get_data_transforms():
    # 训练集数据增强：CIFAR-10 标准增强策略（四周填充 4 像素后随机裁剪 + 水平翻转）
    train_transform = transforms.Compose([
        # 四周填充 4 像素，再随机裁剪回 32×32，等价于随机平移，增强位置鲁棒性
        transforms.RandomCrop(IMG_SIZE, padding=4),
        # 50% 概率水平翻转图片
        transforms.RandomHorizontalFlip(p=0.5),
        # 图片转张量 [0,255] → [0,1]
        transforms.ToTensor(),
        # CIFAR-10 标准化均值方差，统一数据分布（统计自 5 万张训练图）
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    # 验证集不需要增强，只标准化，保证评估公平
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    return train_transform, eval_transform

# 加载、划分数据集，生成数据加载器
def prepare_data():
    train_transform, eval_transform = get_data_transforms()
    # 加载 CIFAR-10 训练集（共 50000 张，已自带标签 0~9）
    full_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False)
    print(f"识别类别: {full_dataset.classes}")
    print(f"训练集总图片数量: {len(full_dataset)}")

    # 计算验证集、训练集样本数量
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    # 随机划分数据集索引
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    # 关键：训练集、验证集使用不同预处理
    train_dataset = Subset(
        datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_transform),
        train_subset.indices
    )
    val_dataset = Subset(
        datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=eval_transform),
        val_subset.indices
    )
    print(f"训练图片数量: {len(train_dataset)} 验证图片数量: {len(val_dataset)}")
    # DataLoader：分批读取数据，shuffle 打乱训练集
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, full_dataset.classes

# 训练一轮的函数
def train_one_epoch(model, loader, criterion, optimizer, device):
    # 开启训练模式：启用 Dropout、BN 训练模式
    model.train()
    total_loss = 0.0  # 累计总损失
    correct = 0        # 预测正确图片数
    samples = 0        # 总图片数
    # 循环读取每一批图片
    for batch_idx, (inputs, labels) in enumerate(loader):
        # 数据放到 GPU/CPU 上
        inputs, labels = inputs.to(device), labels.to(device)
        # 1. 清空上一轮梯度，必须操作
        optimizer.zero_grad()
        # 2. 前向传播，得到模型预测输出
        outputs = model(inputs)
        # 3. 计算损失，对比预测和真实标签
        loss = criterion(outputs, labels)
        # 4. 反向传播，计算梯度
        loss.backward()
        # 5. 优化器更新网络参数
        optimizer.step()

        # 统计预测准确率
        preds = outputs.argmax(dim=1)  # 取概率最大的类别作为预测结果
        total_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        samples += inputs.size(0)

        # 每 50 批打印一次训练信息
        if (batch_idx + 1) % 50 == 0:
            print(f" Batch [{batch_idx+1}/{len(loader)}] Loss: {total_loss/samples:.4f} Acc: {correct/samples:.4f}")
    # 返回本轮平均损失、平均准确率
    return total_loss / samples, correct / samples

# 验证一轮函数，不更新参数，@torch.no_grad 关闭梯度计算省显存
@torch.no_grad()
def validate(model, loader, criterion, device):
    # 评估模式：关闭 Dropout，BN 使用全局均值方差
    model.eval()
    total_loss, correct, samples = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        samples += inputs.size(0)
    return total_loss / samples, correct / samples

# 绘制训练损失、准确率曲线并保存图片
def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve'); ax1.legend(); ax1.grid(alpha=0.3)
    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线图片已保存: {save_path}")

# 主训练函数，程序入口
def train():
    print("=" * 50)
    print(f"训练使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    # 固定随机种子
    set_seed(RANDOM_SEED)
    # 加载数据
    train_loader, val_loader, class_names = prepare_data()
    # 创建模型并放到 GPU/CPU（CIFAR-10 共 10 个类别）
    model = create_model(num_classes=len(class_names)).to(DEVICE)
    # 损失函数：交叉熵损失，多分类专用
    criterion = nn.CrossEntropyLoss()
    # 优化器 Adam，自适应学习率
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 学习率调度器：验证损失连续 4 轮不下降，学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    # 记录每轮训练数据
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0  # 记录最高验证准确率
    t_start = time.time()  # 记录总训练时间

    # 循环训练每一轮
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}] 当前学习率={optimizer.param_groups[0]['lr']:.6f}")
        t0 = time.time()
        # 训练一轮
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        # 验证一轮
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        t1 = time.time()
        # 根据验证损失调整学习率
        scheduler.step(val_loss)
        # 保存本轮数据
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        # 打印本轮结果
        print(f" 训练 Loss: {train_loss:.4f} 准确率: {train_acc:.4f}")
        print(f" 验证 Loss: {val_loss:.4f} 准确率: {val_acc:.4f}")
        print(f" 本轮耗时: {t1-t0:.1f}s")
        # 如果当前验证准确率是历史最高，保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" [✓] 保存最优模型，当前最佳准确率={best_acc:.4f}")
    # 全部轮次训练完成
    print(f"\n训练全部完成! 总耗时 {(time.time()-t_start)/60:.1f}分钟, 最佳验证准确率={best_acc:.4f}")
    # 绘制曲线
    plot_training_history(history, os.path.join(PROJECT_DIR, "training_history.png"))

# 程序入口，运行脚本时执行 train()
if __name__ == "__main__":
    train()
