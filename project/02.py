import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== 1. 超参数与设备配置 ====================
BATCH_SIZE = 64  # 批大小：每次迭代送入模型的样本数量，平衡显存占用与梯度估计稳定性
EPOCHS = 5       # 训练轮数：整个训练集被完整遍历的次数
# 自动检测设备：优先使用NVIDIA GPU加速，无GPU时回退到CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 2. 数据集加载与预处理 ====================
# 定义图像变换流水线，按顺序依次执行
transform = transforms.Compose([
    # 将PIL图像转换为Tensor，同时将像素值从[0, 255]线性映射到[0.0, 1.0]
    transforms.ToTensor(),
    # 使用MNIST数据集的全局均值(0.1307)和标准差(0.3081)进行标准化
    # 使输入数据近似标准正态分布，有效加速模型收敛并提升泛化能力
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练集与测试集
# root: 数据存储根目录；train: True为训练集(6万张)，False为测试集(1万张)
# download=True: 本地无数据时自动下载（若遇404错误需手动下载放置后改为False）
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 创建数据加载器
# shuffle=True: 训练集必须打乱顺序，防止模型记忆样本排列导致过拟合
# shuffle=False: 测试集无需打乱，保证评估结果可复现
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== 3. CNN模型定义 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 特征提取模块：卷积层+激活函数+池化层交替堆叠
        self.features = nn.Sequential(
            # 第1个卷积块：输入1通道灰度图 → 输出16通道特征图
            # kernel_size=3, padding=1 保证输出空间尺寸不变(28×28)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输出形状: [B, 16, 28, 28]
            nn.ReLU(inplace=True),  # ReLU激活引入非线性，inplace节省显存
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2×2最大池化，尺寸减半 → [B, 16, 14, 14]

            # 第2个卷积块：16通道 → 32通道
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输出形状: [B, 32, 14, 14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 再次池化 → [B, 32, 7, 7]
        )

        # 分类头模块：全连接层将高维特征映射到10个类别
        self.classifier = nn.Sequential(
            # 展平后的特征维度 = 32(通道) × 7(高) × 7(宽) = 1568
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)  # 输出10维logits（原始分数），CrossEntropyLoss内部会自动执行Softmax
        )

    def forward(self, x):
        """前向传播：定义数据从输入到输出的完整计算图"""
        x = self.features(x)          # 提取空间层次特征
        x = x.view(x.size(0), -1)     # 展平操作：保留batch维度(B)，其余维度合并为一维向量[B, 1568]
        x = self.classifier(x)        # 通过全连接层得到分类结果
        return x

# 实例化模型并移至指定设备
model = SimpleCNN().to(DEVICE)
# 交叉熵损失函数：适用于多分类任务，结合了LogSoftmax与NLLLoss
criterion = nn.CrossEntropyLoss()
# Adam优化器：自适应学习率算法，lr=0.001为经验默认值
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==================== 4. 训练与评估函数 ====================
def train(epoch):
    """单轮训练函数"""
    model.train()  # 【关键】切换至训练模式：启用Dropout/BatchNorm的训练态行为
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移至计算设备(GPU/CPU)
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()                     # 【关键】清零梯度：PyTorch默认累积梯度，不清零会导致梯度爆炸
        output = model(data)                      # 前向传播：构建计算图并得到预测logits
        loss = criterion(output, target)           # 计算当前batch的损失值
        loss.backward()                            # 反向传播：自动微分计算所有可学习参数的梯度
        optimizer.step()                           # 参数更新：根据梯度和学习率调整模型权重

        running_loss += loss.item()
        # 每200个batch打印一次训练状态
        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")

def test():
    """模型评估函数"""
    model.eval()  # 【关键】切换至评估模式：关闭Dropout，BatchNorm使用全局统计量
    test_loss = 0.0
    correct = 0
    # 【关键】评估时禁用梯度计算：大幅减少显存消耗并加速推理
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            # argmax取概率最大索引作为预测类别，keepdim保持维度便于比较
            pred = output.argmax(dim=1, keepdim=True)
            # eq逐元素比较，sum统计正确预测的样本总数
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# ==================== 5. 主训练循环 ====================
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(epoch)   # 执行一轮训练
        test()         # 每轮结束后立即评估，实时监控过拟合与收敛趋势