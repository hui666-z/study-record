import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ================== 配置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5

# 本地预训练权重路径（请确保该文件存在）
PRETRAINED_WEIGHT_PATH = 'model/resnet50-0676ba61.pth'

# 训练后保存的模型路径
MODEL_SAVE_PATH = 'resnet50_pneumonia.pth'

# ================== 数据预处理 ==================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================== 数据集加载（仅 train/test） ==================
class_names = ['Normal', 'Pneumonia']
DATA_ROOT = './data/chest_xray'

train_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/train',
    transform=train_transform
)
test_dataset = datasets.ImageFolder(
    root=f'{DATA_ROOT}/test',
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================== 模型创建函数（使用本地权重） ==================
def create_model(mode="freeze_all"):
    # 1. 创建未预训练的空模型
    model = models.resnet50(weights=None)

    # 2. 加载本地预训练权重
    if not os.path.exists(PRETRAINED_WEIGHT_PATH):
        raise FileNotFoundError(
            f"本地预训练权重文件不存在: {PRETRAINED_WEIGHT_PATH}\n"
            "请从 https://download.pytorch.org/models/resnet50-0676ba61.pth 下载并放置到该位置。"
        )
    state_dict = torch.load(PRETRAINED_WEIGHT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"成功加载本地预训练权重: {PRETRAINED_WEIGHT_PATH}")

    # 3. 根据迁移模式替换分类头并设置优化器
    if mode == "freeze_all":
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    elif mode == "fine_tune":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    else:
        raise ValueError("mode must be 'freeze_all' or 'fine_tune'")
    return model, optimizer

# ================== 训练函数 ==================
def train_model(model, optimizer, criterion, train_loader, epochs, device):
    history = {'train_loss': []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f}")
    return history

# ================== 测试函数 ==================
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    print(f"\n测试集准确率: {correct}/{total} ({accuracy:.2f}%)")
    return accuracy

# ================== 预测可视化函数 ==================
def imshow_tensor(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)

# ================== 主程序 ==================
if __name__ == "__main__":
    # 创建模型（自动加载本地预训练权重）
    model, optimizer = create_model(mode="freeze_all")
    criterion = nn.CrossEntropyLoss()

    # 检查是否有已保存的完整模型（训练后的）
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"发现已保存的完整模型 '{MODEL_SAVE_PATH}'，正在加载...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        print("模型加载成功！")
    else:
        print(f"未找到完整模型文件，开始训练...")
        model = model.to(DEVICE)
        history = train_model(model, optimizer, criterion, train_loader, EPOCHS, DEVICE)

        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': EPOCHS,
        }, MODEL_SAVE_PATH)
        print(f"模型已保存至 '{MODEL_SAVE_PATH}'")

        # 绘制训练损失曲线
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss', marker='o')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 在测试集上评估
    print("\n===== 在测试集上评估模型 =====")
    test_accuracy = test_model(model, test_loader, DEVICE)

    # 可视化部分测试样本的预测结果
    images, labels = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(DEVICE))
        preds = outputs.argmax(dim=1).cpu()

    plt.figure(figsize=(10, 10))
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        ax.axis('off')
        pred_txt = class_names[preds[i]]
        true_txt = class_names[labels[i]]
        title_text = f"预测: {pred_txt} | 真实: {true_txt}"
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(title_text, color=color, fontsize=12)
        imshow_tensor(images[i])
    plt.show()