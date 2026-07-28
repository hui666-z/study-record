#python C:\Users\92647\Desktop\coding\test.p.c++.web\python\project\01.py
# 导入所需依赖库
import torch
# 神经网络基础模块（层、激活函数、容器等）
import torch.nn as nn
# 优化器模块（Adam、SGD等参数更新算法）
import torch.optim as optim
# 计算机视觉数据集、图像预处理工具
from torchvision import datasets, transforms
# 数据加载器，分批次迭代数据集
from torch.utils.data import DataLoader
# 绘图可视化库，绘制训练曲线与预测样本
import matplotlib.pyplot as plt
# 数值计算库，张量转图像时使用
import numpy as np

# ==================== 0. 全局超参数配置 ====================
# 设置matplotlib中文显示字体（Windows黑体SimHei），解决图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决图表负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 自动判断可用设备：有NVIDIA显卡用CUDA，无则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 批次大小：单次送入网络64张图片计算梯度
BATCH_SIZE = 64
# 训练轮次：完整遍历全部训练集5次
EPOCHS = 5
# 学习率：Adam优化器的参数更新步长
LR = 0.001

# 打印当前使用的计算设备，确认GPU/CPU
print(f"当前设备: {DEVICE}")
# 打印PyTorch版本，方便环境复现与排错
print(f"PyTorch版本: {torch.__version__}")

# ==================== 1. 数据加载与预处理 ====================
# 组合图像预处理流水线，按顺序执行操作
transform = transforms.Compose([
    # 将PIL图像/数组转为张量，像素值从0~255映射至0~1，通道顺序HWC转为CHW
    transforms.ToTensor(),
    # MNIST数据集专用标准化：(均值, 标准差)，加速模型收敛、稳定训练
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST手写数字训练集，不存在则自动下载，绑定预处理流水线
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# 加载MNIST手写数字测试集，用于验证模型泛化能力
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 训练集加载器：按BATCH_SIZE分批次，shuffle=True打乱样本防止模型记忆顺序
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 测试集加载器：不打乱顺序，保证每次验证结果可复现
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 取出一个批次样本，校验数据维度是否符合预期
sample_imgs, sample_labels = next(iter(train_loader))
print(f"\nBatch形状: {sample_imgs.shape}")  # 预期输出 [批次大小, 通道数, 高, 宽] → [64,1,28,28]
print(f"标签范围: {sample_labels.min()}~{sample_labels.max()}")  # MNIST标签为0~9数字

# ==================== 2. MLP多层感知机模型定义 ====================
# 自定义全连接网络类，继承PyTorch基础模型nn.Module
class MLP(nn.Module):
    # 网络层构造函数：定义所有可训练层
    def __init__(self):
        super().__init__()  # 调用父类构造，注册网络子模块
        # Sequential容器：有序串联网络层，简化前向传播代码
        self.net = nn.Sequential(
            # 展平层：把[B,1,28,28]四维图像转为[B,784]一维向量，适配全连接层输入
            nn.Flatten(),
            # 全连接层1：输入784维像素向量，输出256维隐藏特征
            nn.Linear(784, 256),
            # ReLU激活函数，引入非线性表达；inplace=True原地运算节省显存
            nn.ReLU(inplace=True),
            # 全连接层2：256维隐藏特征压缩至128维
            nn.Linear(256, 128),
            # 第二层非线性激活
            nn.ReLU(inplace=True),
            # 输出层：128维特征映射为10个类别的原始得分(logits)，不添加softmax
            nn.Linear(128, 10)
        )

    # 前向传播函数：定义数据流过网络的计算逻辑，PyTorch自动构建梯度计算图
    def forward(self, x):
        return self.net(x)  # 输入图像依次经过所有层，返回分类原始得分

# 实例化模型，并将模型权重迁移至指定计算设备(GPU/CPU)
model = MLP().to(DEVICE)

# 统计模型所有可训练参数总数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n可训练参数量: {total_params:,}")
# 打印完整网络层级结构，用于核对模型搭建是否正确
print(model)

# ==================== 3. 损失函数与优化器 ====================
# 交叉熵损失函数：内部集成LogSoftmax+负对数损失，输入无需手动softmax
criterion = nn.CrossEntropyLoss()
# Adam自适应学习率优化器，传入模型可训练参数与初始学习率
optimizer = optim.Adam(model.parameters(), lr=LR)

# 打印损失函数类型，提醒输出层无需激活
print(f"\n损失函数: CrossEntropyLoss")
# 打印优化器配置信息
print(f"优化器: Adam (lr={LR})")

# ==================== 4. 训练与验证循环 ====================
# 训练记录字典：保存每轮平均训练损失、测试集准确率，用于后续绘图
history = {'train_loss': [], 'val_acc': []}

# 遍历每一轮完整训练
for epoch in range(1, EPOCHS + 1):
    # --- 训练阶段 ---
    model.train()  # 切换为训练模式：启用Dropout、BatchNorm参数更新逻辑
    running_loss = 0.0  # 累加当前轮次全部样本损失
    # 迭代遍历训练集所有批次数据
    for imgs, labels in train_loader:
        # 将图像、标签同步迁移至GPU/CPU设备
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()       # Step1：清空上一批次累积梯度，避免梯度叠加
        outputs = model(imgs)       # Step2：前向传播，输入图像得到10分类原始得分
        loss = criterion(outputs, labels)  # Step3：计算当前批次预测与真实标签的损失
        loss.backward()             # Step4：反向传播，链式求导计算各参数梯度
        optimizer.step()            # Step5：根据梯度更新模型权重参数

        # 累加加权损失（单样本损失×批次样本数量）
        running_loss += loss.item() * imgs.size(0)
    # 计算本轮所有训练样本的平均损失
    avg_train_loss = running_loss / len(train_loader.dataset)

    # --- 验证阶段 ---
    model.eval()  # 切换为评估模式：冻结BatchNorm、关闭Dropout，仅做推理
    correct = 0   # 统计测试集预测正确样本总数
    # torch.no_grad()上下文：禁用梯度计算，减少显存占用、加速推理
    with torch.no_grad():
        # 迭代遍历测试集所有批次
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # argmax取得分最高维度索引，作为预测类别
            preds = model(imgs).argmax(dim=1)
            # 统计本批次预测与标签相等的样本数量，累加入总正确数
            correct += (preds == labels).sum().item()
    # 计算测试集整体分类准确率（百分比）
    val_acc = 100.0 * correct / len(test_loader.dataset)

    # 将本轮指标存入历史记录，用于绘图
    history['train_loss'].append(avg_train_loss)
    history['val_acc'].append(val_acc)
    # 打印本轮训练日志：轮次、平均损失、测试准确率
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ==================== 5. 训练曲线可视化 ====================
# 创建画布，1行2列子图，设置画布尺寸
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：绘制训练损失变化曲线，蓝色圆点实线
ax1.plot(range(1, EPOCHS+1), history['train_loss'], 'bo-', linewidth=2, markersize=8)
ax1.set_title('训练损失曲线', fontsize=14)  # 子图标题
ax1.set_xlabel('Epoch')                     # X轴标签：训练轮次
ax1.set_ylabel('Loss')                      # Y轴标签：损失值
ax1.grid(True, alpha=0.3)                   # 显示浅色网格，方便读数

# 右图：绘制验证准确率变化曲线，绿色三角实线
ax2.plot(range(1, EPOCHS+1), history['val_acc'], 'g^-', linewidth=2, markersize=8)
ax2.set_title('验证准确率曲线', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

# 自动调整子图间距，防止标题、坐标轴文字重叠
plt.tight_layout()
# 弹出窗口展示训练曲线
plt.show()

# ==================== 6. 测试集预测结果可视化 ====================
# 反标准化函数：将标准化后的张量还原至0~1原始像素区间，用于绘图
def denormalize(img_tensor):
    return img_tensor * 0.3081 + 0.1307

model.eval()  # 推理模式，禁止参数更新
# 取出测试集一个批次样本用于可视化
vis_imgs, vis_labels = next(iter(test_loader))
# 无梯度推理，得到预测类别并迁移回CPU（绘图仅支持CPU数组）
with torch.no_grad():
    vis_preds = model(vis_imgs.to(DEVICE)).argmax(dim=1).cpu()

# 创建2行4列子图画布，展示8张手写数字样本
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
# 扁平化遍历所有子图坐标轴
for i, ax in enumerate(axes.flatten()):
    ax.axis('off')  # 关闭坐标轴刻度，只显示图像
    # 反标准化 → 去除通道维度 → 转为numpy数组，适配imshow绘图
    img_np = denormalize(vis_imgs[i]).squeeze().numpy()
    ax.imshow(img_np, cmap='gray')  # 灰度图显示手写数字
    # 判断当前样本预测是否正确
    is_correct = vis_preds[i] == vis_labels[i]
    color = 'green' if is_correct else 'red'  # 正确标题绿色，错误红色
    symbol = '✅' if is_correct else '❌'      # 匹配对错标识符号
    # 设置子图标题：标注真实标签、预测标签
    ax.set_title(f'{symbol} 真:{vis_labels[i]} 预:{vis_preds[i]}', color=color, fontsize=12)

# 设置整张画布总标题
plt.suptitle('测试集预测样例（绿=正确 / 红=错误）', fontsize=15)
plt.tight_layout()
# 弹出窗口展示预测样本效果图
plt.show()