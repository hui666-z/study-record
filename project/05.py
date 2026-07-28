# ---------------------- 导入需要的工具包 ----------------------
# pytorch核心库，搭建神经网络
import torch
# 网络层、损失函数
import torch.nn as nn
# 优化器（用来更新模型参数）
import torch.optim as optim
# 预训练网络、图片处理、读取文件夹图片
from torchvision import models, transforms, datasets
# 分批读取数据
from torch.utils.data import DataLoader
# 画图工具，展示图片、曲线
import matplotlib.pyplot as plt
# 数值计算库
import numpy as np
# 文件路径操作
import os

# ===================== 基础设置（初学者直接照抄） =====================
# 让matplotlib正常显示中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号乱码
plt.rcParams['axes.unicode_minus'] = False

# 自动判断有没有显卡GPU
# 有GPU就用GPU加速训练，没有就用CPU慢慢跑
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32    # 一次性放进网络多少张图片
EPOCHS = 5         # 完整遍历整个训练集多少次

# ResNet50预训练权重文件路径（提前下载放到对应文件夹）
PRETRAINED_WEIGHT_PATH = 'model/resnet50-0676ba61.pth'
# 训练完成后，保存模型的文件名
MODEL_SAVE_PATH = 'resnet50_pneumonia.pth'

# ===================== 图片预处理 =====================
# 训练集图片处理规则（加入随机翻转，扩充数据，防止模型死记硬背）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 统一把图片缩放到 224×224
    transforms.RandomHorizontalFlip(),      # 随机左右翻转图片
    transforms.ToTensor(),                  # 图片转为神经网络能识别的格式
    # 标准化（必须和预训练模型保持一致！固定参数，不用修改）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集图片处理规则（测试时不能随机翻转图片！）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 读取数据集 =====================
# 两类：正常胸片、肺炎胸片
class_names = ['Normal', 'Pneumonia']
# 数据集总文件夹
DATA_ROOT = './data/chest_xray'

# ImageFolder规则：文件夹名字=类别名，自动给图片打上标签
train_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/train',
    transform=train_transform
)
test_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/test',
    transform=test_transform
)

# DataLoader：自动分批、打乱、加载图片
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 创建ResNet50迁移学习模型 =====================
# 两种模式：
# freeze_all：冻结网络主干，只训练最后一层（简单、训练快，新手推荐先跑这个）
# fine_tune：全部网络一起微调（效果更好、训练更慢）
def create_model(mode="freeze_all"):
    # 创建空白的ResNet50网络结构，暂时没有权重
    model = models.resnet50(weights=None)

    # 判断本地权重文件是否存在，不存在直接报错提醒你下载
    if not os.path.exists(PRETRAINED_WEIGHT_PATH):
        raise FileNotFoundError(
            f"找不到权重文件: {PRETRAINED_WEIGHT_PATH}\n"
            "去官网下载 resnet50-0676ba61.pth 放到model文件夹！"
        )
    # 加载网上训练好的权重（图像通用特征）
    state_dict = torch.load(PRETRAINED_WEIGHT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"成功加载预训练权重")

    if mode == "freeze_all":
        # 冻结所有层：主干网络参数不更新，只使用它提取图片特征
        for param in model.parameters():
            param.requires_grad = False
        # 获取网络最后一层的输入维度
        num_ftrs = model.fc.in_features
        # 替换最后一层，改成2分类（正常/肺炎）
        model.fc = nn.Linear(num_ftrs, 2)
        # 只更新最后一层参数，学习率大一点
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    elif mode == "fine_tune":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        # 全部参数一起训练，学习率要调小，避免破坏原有特征
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    else:
        raise ValueError("mode只能填 freeze_all 或者 fine_tune")
    return model, optimizer

# ===================== 训练一轮的函数 =====================
def train_model(model, optimizer, criterion, train_loader, epochs, device):
    # 用来保存每一轮损失，之后画图
    train_loss_history = []

    # 循环训练多少轮
    for epoch in range(1, epochs + 1):
        model.train()  # 开启训练模式
        total_loss = 0.0

        # 循环读取每一批图片
        for images, labels in train_loader:
            # 把图片、标签放到GPU/CPU上
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()      # 清空上一轮的梯度（必须写！）
            outputs = model(images)    # 图片传入网络，得到预测结果
            loss = criterion(outputs, labels)  # 计算预测和真实标签差距（损失）
            loss.backward()            # 反向传播，计算梯度
            optimizer.step()           # 根据梯度更新网络参数

            # 累加损失
            total_loss = total_loss + loss.item() * images.size(0)

        # 计算这一轮整张数据集的平均损失
        avg_loss = total_loss / len(train_loader.dataset)
        train_loss_history.append(avg_loss)
        print(f"第 {epoch} 轮训练，平均损失值：{avg_loss:.4f}")

    return train_loss_history

# ===================== 测试函数（计算准确率） =====================
def test_model(model, test_loader, device):
    model.eval()  # 测试模式，关闭随机操作
    correct_num = 0  # 预测正确数量
    total_num = 0     # 总共图片数量

    # 不计算梯度，节省内存、提速
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 找到得分最高的类别作为预测结果
            predict = outputs.argmax(dim=1)
            # 统计预测正确样本
            correct_num = correct_num + (predict == labels).sum().item()
            total_num = total_num + labels.size(0)

    acc = 100.0 * correct_num / total_num
    print(f"\n测试集：一共{total_num}张，预测正确{correct_num}张，准确率 {acc:.2f} %")
    return acc

# ===================== 工具函数：展示张量图片 =====================
def imshow_tensor(inp):
    # 把神经网络格式图片转回plt可以显示的格式
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 逆标准化，还原图片原始亮度色彩
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

# ===================== 程序入口（从这里开始运行） =====================
if __name__ == "__main__":
    # 创建模型与优化器，新手先用 freeze_all
    model, optimizer = create_model(mode="freeze_all")
    # 分类任务损失函数
    criterion = nn.CrossEntropyLoss()

    # 判断本地有没有已经训练好的模型
    if os.path.exists(MODEL_SAVE_PATH):
        print("检测到已有训练完成的模型，直接加载模型，跳过训练！")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
    else:
        print("没有找到模型文件，开始全新训练！")
        model = model.to(DEVICE)
        loss_record = train_model(model, optimizer, criterion, train_loader, EPOCHS, DEVICE)

        # 保存训练好的模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, MODEL_SAVE_PATH)
        print("模型保存完成！")

        # 绘制训练损失曲线
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, EPOCHS+1), loss_record, marker='o')
        plt.title("训练损失变化曲线")
        plt.xlabel("训练轮数 Epoch")
        plt.ylabel("损失 Loss")
        plt.grid(True)
        plt.show()

    # 运行测试，输出准确率
    print("\n========== 开始测试模型 ==========")
    test_model(model, test_loader, DEVICE)

    # 随机拿一批图片，可视化预测效果
    images, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        out = model(images.to(DEVICE))
        pred = out.argmax(dim=1).cpu()

    # 画2行2列一共4张图
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.axis("off")
        real_label = class_names[labels[i]]
        pred_label = class_names[pred[i]]
        # 预测正确标题绿色，错误红色
        if real_label == pred_label:
            plt.title(f"预测:{pred_label} 真实:{real_label}", color="green")
        else:
            plt.title(f"预测:{pred_label} 真实:{real_label}", color="red")
        imshow_tensor(images[i])
    plt.show()