"""
单张图片猫狗预测工具
使用命令：python predict.py 图片路径.jpg
"""
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import create_model

# 配置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "cat_dog_cnn_best.pth")
IMG_SIZE = 224
CLASS_NAMES = ['猫(Cat)', '狗(Dog)']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取图片并预处理
def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 统一转为RGB，兼容png透明图、灰度图
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0) # 增加batch维度 [C,H,W] → [1,C,H,W]
    return img, tensor

# 预测函数
def predict_img(img_path, model):
    img, tensor = load_image(img_path)
    tensor = tensor.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    return pred_idx.item(), conf.item(), probs.squeeze().cpu().numpy(), img

# 主程序入口
def main():
    # 判断是否传入图片路径参数
    if len(sys.argv) < 2:
        print("使用方法：python predict.py 你的图片路径")
        print("示例：python predict.py data/test_set/test_set/cats/cat.4001.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    # 判断文件存在
    if not os.path.exists(img_path):
        print(f"错误：图片文件不存在 {img_path}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在，请先运行train.py训练！")
        sys.exit(1)
    # 加载模型
    model = create_model(pretrained_path=MODEL_PATH).to(DEVICE)
    # 预测
    pred_idx, conf, prob_arr, raw_img = predict_img(img_path, model)
    # 打印文字结果
    print(f"\n{'='*40}")
    print(f"图片名称: {os.path.basename(img_path)}")
    print(f"预测结果: {CLASS_NAMES[pred_idx]}，置信度: {conf:.4f}")
    print(f"{'='*40}")
    # 打印概率进度条
    for idx, name in enumerate(CLASS_NAMES):
        bar = '#' * int(prob_arr[idx] * 40) + '-' * (40 - int(prob_arr[idx] * 40))
        mark = " <-- 预测结果" if idx == pred_idx else ""
        print(f"{name}: {bar} {prob_arr[idx]:.4f} ({prob_arr[idx]*100:.1f}%){mark}")

    # 绘制图片+概率柱状图并保存
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(raw_img)
    ax1.set_title(os.path.basename(img_path))
    ax1.axis('off')
    # 柱状图
    bar_color = ['#FF6B6B' if i == pred_idx else '#95E1D3' for i in range(2)]
    bars = ax2.bar(CLASS_NAMES, prob_arr, color=bar_color)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("预测概率")
    ax2.set_title(f"最终判定：{CLASS_NAMES[pred_idx]} {conf:.2%}")
    # 柱子上标注数值
    for bar, p in zip(bars, prob_arr):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{p:.4f}", ha="center")
    plt.tight_layout()
    save_name = f"predict_{os.path.basename(img_path)}"
    save_path = os.path.join(PROJECT_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    print(f"\n预测可视化图已保存: {save_path}")

if __name__ == "__main__":
    main()