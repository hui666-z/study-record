"""
单张图片预测脚本（适配 CIFAR-10）
使用方法：python predict.py 图片路径
示例：python predict.py data/cifar-10-batches-py/test_batch 中的任意图片
说明：传入任意图片，脚本会自动缩放至 32×32 并输出 CIFAR-10 十大类别预测结果
"""
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import create_model

# 基础配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "cifar10_cnn_best.pth")
IMG_SIZE = 32
# CIFAR-10 十个类别名称（中英文对照）
CLASS_NAMES = [
    '飞机 (airplane)', '汽车 (automobile)', '鸟 (bird)', '猫 (cat)',
    '鹿 (deer)', '狗 (dog)', '青蛙 (frog)', '马 (horse)',
    '船 (ship)', '卡车 (truck)'
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取图片并预处理
def load_image(path):
    transform = transforms.Compose([
        # 统一缩放到 32×32，与训练尺寸一致
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    # 打开图片，统一转为 RGB（兼容 png 透明图、灰度图）
    image = Image.open(path).convert('RGB')
    # 增加 batch 维度 [3,32,32] → [1,3,32,32]
    tensor = transform(image).unsqueeze(0)
    return image, tensor

# 预测核心函数
def predict(path, model):
    image, tensor = load_image(path)
    tensor = tensor.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        # softmax 转为概率
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return pred.item(), conf.item(), probs.squeeze().cpu().numpy(), image

# 程序入口
def main():
    # 判断是否传入图片路径
    if len(sys.argv) < 2:
        print("使用命令格式：python predict.py 你的图片路径")
        print("示例：python predict.py test.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    # 判断图片是否存在
    if not os.path.exists(img_path):
        print(f"错误：图片文件不存在 {img_path}")
        sys.exit(1)
    # 判断模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：未找到训练模型 {MODEL_PATH}，请先运行 train.py 训练！")
        sys.exit(1)
    # 加载模型（10 个类别）
    model = create_model(num_classes=len(CLASS_NAMES), pretrained_path=MODEL_PATH).to(DEVICE)
    # 执行预测
    pred, conf, probs, img = predict(img_path, model)
    # 打印文字结果
    print(f"\n{'='*40}")
    print(f"图片文件名: {os.path.basename(img_path)}")
    print(f"预测结果: {CLASS_NAMES[pred]} 置信度: {conf:.4f}")
    print(f"{'='*40}")
    # 打印概率条形文本
    for i, name in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 40) + '-' * (40 - int(probs[i] * 40))
        mark = " <--本次预测结果" if i == pred else ""
        print(f"{name:20s}: {bar} {probs[i]:.4f} ({probs[i]*100:.1f}%){mark}")
    # 绘制原图+概率柱状图并保存
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(img)
    ax1.set_title(os.path.basename(img_path))
    ax1.axis('off')
    colors = ['#FF6B6B' if i == pred else '#95E1D3' for i in range(len(CLASS_NAMES))]
    # 只取中文名用于柱状图显示，避免标签过长
    short_names = [name.split(' ')[0] for name in CLASS_NAMES]
    bars = ax2.bar(short_names, probs, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("预测概率")
    ax2.set_title(f"预测结论: {CLASS_NAMES[pred]} ({conf:.2%})")
    # x 轴标签旋转 45 度防重叠
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for bar, p in zip(bars, probs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{p:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    save_name = f"predict_{os.path.basename(img_path)}"
    save_path = os.path.join(PROJECT_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n预测可视化图片已保存至: {save_path}")

if __name__ == "__main__":
    main()
