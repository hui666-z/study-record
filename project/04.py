import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== 1. 超参数与设备配置 ====================
BATCH_SIZE = 64  # 批大小：ResNet含BN层，batch过小会导致统计量估计不准
EPOCHS = 5       # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 2. MNIST数据准备 ====================
transform = transforms.Compose([
    transforms.ToTensor(),                  # PIL→Tensor, [0,255]→[0,1], HWC→CHW
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST全局均值/标准差标准化
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 训练集shuffle防过拟合；测试集不shuffle保证评估可复现
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 3. ResNet基础残差块 (BasicBlock) ====================
class ResidualBlock(nn.Module):
    """
    标准ResNet BasicBlock结构:
    x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+shortcut) → ReLU → out

    核心公式: y = F(x) + x
    - F(x): 两层卷积学到的残差映射
    - x: 恒等映射（shortcut），使梯度可直接回传至浅层
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径：两个3×3卷积
        # bias=False: BN层已有可学习偏移量β，卷积bias冗余且浪费参数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)         # 批归一化：加速收敛+正则化
        self.relu = nn.ReLU(inplace=True)               # inplace节省显存
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径路径（Shortcut / Skip Connection）
        # 仅当空间尺寸(stride≠1)或通道数(in≠out)不匹配时才添加1×1卷积对齐
        self.shortcut = nn.Sequential()  # 默认恒等映射（无参数）
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1×1卷积：调整通道数；stride同步主路径的空间下采样
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)  # shortcut也需BN保持分布一致
            )

    def forward(self, x):
        identity = x  # 保存输入用于残差相加
        out = self.relu(self.bn1(self.conv1(x)))       # Conv→BN→ReLU
        out = self.bn2(self.conv2(out))                 # Conv→BN（注意：此处不加ReLU）
        out += self.shortcut(identity)                  # 核心：残差相加 F(x)+x
        out = self.relu(out)                            # 相加后再激活，避免ReLU截断负值信息
        return out

# ==================== 4. 适配MNIST的MiniResNet-18 ====================
class MiniResNet18(nn.Module):
    """
    相比标准ResNet-18的关键改造：
    1. 输入通道: 3(RGB) → 1(灰度)
    2. 初始卷积: 7×7/stride=2 → 3×3/stride=1（MNIST仅28×28，7×7会丢失过多信息）
    3. 移除MaxPool: 避免小图过早下采样至无效尺寸
    4. 层数缩减: 仅保留3个stage（标准版为4个），防止对小数据集严重过拟合
    """
    def __init__(self, num_classes=10):
        super(MiniResNet18, self).__init__()
        self.in_channels = 64  # 动态追踪当前特征图通道数

        # 预处理层：替代标准ResNet的Conv7x7+MaxPool
        self.prep = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 输出: [B, 64, 28, 28]

        # 残差阶段堆叠（每个stage含2个BasicBlock）
        self.layer1 = self._make_layer(64, stride=1)   # [B, 64, 28, 28] 尺寸不变
        self.layer2 = self._make_layer(128, stride=2)  # [B, 128, 14, 14] 下采样2倍
        self.layer3 = self._make_layer(256, stride=2)  # [B, 256, 7, 7]   下采样2倍

        # 自适应全局平均池化：无论输入尺寸如何，输出固定为1×1
        # 比Flatten后接FC更鲁棒，且大幅减少参数量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 7, 7] → [B, 256, 1, 1]
        self.fc = nn.Linear(256, num_classes)        # 分类头

    def _make_layer(self, out_channels, stride):
        """构建一个残差阶段：第一个block可能下采样，第二个block保持尺寸"""
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels  # 更新通道数供下一个block使用
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)        # [B,1,28,28] → [B,64,28,28]
        x = self.layer1(x)      # [B,64,28,28]
        x = self.layer2(x)      # [B,128,14,14]
        x = self.layer3(x)      # [B,256,7,7]
        x = self.avgpool(x)     # [B,256,1,1]
        x = torch.flatten(x, start_dim=1)  # [B,256] 安全展平
        x = self.fc(x)          # [B,10] logits
        return x

# ==================== 5. 模型实例化与优化器配置 ====================
model = MiniResNet18(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()               # 内含Softmax，接收原始logits
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam自适应学习率

# ==================== 6. 训练与测试函数 ====================
def train(epoch):
    """单轮训练：启用BN训练态 + Dropout（如有）"""
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()               # 清零累积梯度
        output = model(data)                # 前向传播
        loss = criterion(output, target)
        loss.backward()                     # 反向传播
        optimizer.step()                    # 参数更新
        train_loss += loss.item()
        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")

def test():
    """模型评估：切换eval模式 + 禁用梯度计算"""
    model.eval()  # BN使用全局统计量
    test_loss = 0.0
    correct = 0
    with torch.no_grad():  # 评估时无需梯度
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# ==================== 7. 主训练循环 ====================
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test()