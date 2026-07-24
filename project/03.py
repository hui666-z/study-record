import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== 设备配置 ====================
# 自动检测GPU，优先使用CUDA加速训练，无GPU时回退CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 1. 数据准备 (CIFAR-10 彩色三通道) ====================
# CIFAR-10专用标准化参数（RGB三通道各自的均值与标准差）
# 注意：这些值由CIFAR-10训练集统计得出，不可与MNIST参数混用
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL→Tensor，像素值[0,255]→[0.0,1.0]，同时HWC→CHW
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # R/G/B三通道均值
                         (0.2023, 0.1994, 0.2010))  # R/G/B三通道标准差
])

# 加载CIFAR-10数据集（5万张训练图 + 1万张测试图，32×32×3彩色图像，10个类别）
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 创建数据加载器：batch_size=128适配GPU显存，训练集shuffle防过拟合
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==================== 2. 改进型CNN模型定义 ====================
class CIFAR10CNN(nn.Module):
    """
    相比基础CNN的改进点：
    1. 每个卷积层后接BatchNorm2d，加速收敛并允许更高学习率
    2. 池化层后接Dropout(0.25)，抑制特征提取阶段的过拟合
    3. 全连接层使用BatchNorm1d + Dropout(0.5)，强化分类头正则化
    """
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # 特征提取模块
        self.features = nn.Sequential(
            # --- 第1组卷积块 ---
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B,3,32,32] → [B,32,32,32]
            nn.BatchNorm2d(32),  # 对32个特征图逐通道归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B,32,32,32] → [B,64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B,64,32,32] → [B,64,16,16]
            nn.Dropout(0.25),  # 随机丢弃25%神经元，防过拟合

            # --- 第2组卷积块 ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B,64,16,16] → [B,128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B,128,16,16] → [B,128,8,8]
            nn.Dropout(0.25)
        )
        # 分类头模块
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),  # 展平维度: 128×8×8 = 8192
            nn.BatchNorm1d(512),  # 全连接层后的1D批归一化
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 分类头丢弃率通常高于特征层
            nn.Linear(512, 10)  # 输出10类logits
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 动态展平，兼容任意batch size
        x = self.classifier(x)
        return x

# 实例化模型、损失函数、优化器与学习率调度器
model = CIFAR10CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵（内含Softmax）
optimizer = optim.Adam(model.parameters(), lr=0.001)
# StepLR: 每10个epoch将学习率乘以gamma=0.5，后期精细调优防止震荡
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ==================== 3. 训练与评估函数 ====================
def train_epoch():
    """单轮训练"""
    model.train()  # 启用BN的训练态(使用batch统计量) + Dropout激活
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()  # 清零累积梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

def evaluate(epoch):
    """
    评估函数：除计算准确率外，返回预测值与真实值列表
    用于后续构建混淆矩阵、计算Precision/Recall/F1等细粒度指标
    """
    model.eval()  # 启用BN的推理态(使用全局统计量) + 关闭Dropout
    all_preds = []
    all_targets = []
    with torch.no_grad():  # 评估时禁用梯度，节省显存
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            preds = output.argmax(dim=1)
            # 移至CPU转numpy，避免GPU内存持续占用
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc = 100. * correct / len(all_targets)
    print(f"Epoch {epoch} Test Accuracy: {acc:.2f}%")
    return all_preds, all_targets  # 返回原始数组供混淆矩阵使用

# ==================== 4. 主训练循环 ====================
if __name__ == '__main__':
    for epoch in range(1, 11):
        train_epoch()
        preds, targets = evaluate(epoch)
        scheduler.step()  # ⚠ StepLR按epoch步进，放在evaluate之后

    # 扩展用法示例（取消注释即可生成混淆矩阵）：
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # import matplotlib.pyplot as plt
    # cm = confusion_matrix(targets, preds)
    # ConfusionMatrixDisplay(cm).plot()
    # plt.show()