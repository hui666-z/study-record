"""
猫狗识别 - CNN卷积神经网络模型定义
初学者注释版：搭建5层卷积+全局平均池化+三层全连接分类网络
"""
# 导入pytorch基础包
import torch
import torch.nn as nn

# 自定义CNN网络类，继承nn.Module（所有模型的基类）
class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2):
        # 必须调用父类初始化
        super(CatDogCNN, self).__init__()

        # ========== 卷积块1：输入3通道RGB → 输出32通道，图片尺寸224→112 ==========
        self.conv1 = nn.Sequential(
            # 3×3卷积，padding=1保证输出尺寸不变
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            # 批量归一化，加速训练、防止梯度消失
            nn.BatchNorm2d(32),
            # 激活函数inplace=True原地计算，节省显存
            nn.ReLU(inplace=True),
            # 第二层3×3卷积，提升非线性能力
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 最大池化，尺寸减半(224→112)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷积层dropout，随机屏蔽10%特征通道，防过拟合
            nn.Dropout2d(0.1)
        )

        # 复用_make_block函数快速构建剩余卷积块，通道数翻倍，dropout逐步提高
        self.conv2 = self._make_block(in_ch=32, out_ch=64, dropout=0.15)  # 112→56
        self.conv3 = self._make_block(in_ch=64, out_ch=128, dropout=0.20) # 56→28
        self.conv4 = self._make_block(in_ch=128, out_ch=256, dropout=0.25)# 28→14
        self.conv5 = self._make_block(in_ch=256, out_ch=512, dropout=0.30)# 14→7

        # 全局平均池化：把512×7×7特征图压缩成512×1×1，大幅减少全连接参数
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ========== 分类头：三层全连接层输出2分类（猫/狗） ==========
        self.classifier = nn.Sequential(
            # 展平：把512×1×1转为一维512个数字
            nn.Flatten(),
            nn.Linear(512, 256),   # 全连接1
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),       # 全连接高dropout，深层更容易过拟合
            nn.Linear(256, 128),   # 全连接2
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) # 输出2个原始分数logits，不用加softmax
        )

    # 工具函数：统一生成卷积块，减少重复代码
    def _make_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(dropout),
        )

    # 前向传播：定义数据流动路线（必写）
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

# 创建模型工厂函数，支持加载预训练权重
def create_model(num_classes=2, pretrained_path=None):
    # 初始化网络
    model = CatDogCNN(num_classes=num_classes)
    # 如果传入权重路径，加载训练好的参数
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    return model

# 测试代码：单独运行model.py验证网络输出形状
if __name__ == "__main__":
    model = create_model()
    # 模拟一张224×224 RGB图片，batch=1
    test_img = torch.randn(1, 3, 224, 224)
    output = model(test_img)
    print(f"输入图片shape: {test_img.shape}")  # [1,3,224,224]
    print(f"模型输出shape: {output.shape}")    # [1,2] 猫、狗两个分数