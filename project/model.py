"""
CIFAR-10 十分类 CNN 模型文件
作用：搭建卷积神经网络，定义网络层结构、前向传播逻辑
说明：输入为 3×32×32 彩色图像，输出 10 个类别的预测分数
"""
# 导入pytorch核心库
import torch
import torch.nn as nn

# 搭建 CIFAR-10 识别卷积网络类
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        # 继承父类nn.Module，所有网络都要继承这个类
        super(CIFAR10CNN, self).__init__()

        # ====================== 卷积块1 ======================
        # 输入图片：3通道RGB，尺寸32×32
        self.conv1 = nn.Sequential(
            # 卷积层：3输入通道，32输出通道，卷积核3×3，padding=1保持尺寸不变
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # BN层：批量归一化，加速训练、防止梯度消失
            nn.BatchNorm2d(32),
            # 激活函数ReLU，inplace=True节省显存
            nn.ReLU(inplace=True),
            # 第二层卷积，提取更复杂特征
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 最大池化：尺寸减半，32→16，减少计算量
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 通道dropout，随机丢弃10%特征通道，防止过拟合
            nn.Dropout2d(0.1)
        )

        # 复用通用卷积块函数，堆叠多层卷积
        # CIFAR-10 仅 32×32，前 4 个块带池化，最后 1 个块不带池化（避免特征图过小）
        self.conv2 = self._make_block(32, 64, dropout=0.15)       # 16×16 → 8×8
        self.conv3 = self._make_block(64, 128, dropout=0.20)      # 8×8 → 4×4
        self.conv4 = self._make_block(128, 256, dropout=0.25)     # 4×4 → 2×2
        self.conv5 = self._make_block(256, 512, dropout=0.30, pool=False)  # 保持 2×2

        # 全局平均池化：把512通道2×2特征图压缩成512×1×1，大幅减少全连接参数
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类全连接层
        self.classifier = nn.Sequential(
            # Flatten：把多维特征拉成一维向量
            nn.Flatten(),
            # 全连接层：输入512维，输出256维
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # 神经元dropout，丢弃50%神经元，强力防过拟合
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # 最后输出10个值：CIFAR-10的10个类别
            nn.Linear(128, num_classes)
        )

    # 自定义卷积块工具函数，简化重复代码
    def _make_block(self, in_ch, out_ch, dropout, pool=True):
        """
        构建统一卷积块：两层卷积+BN+激活+【可选池化】+通道dropout
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param dropout: 通道丢弃概率
        :param pool: 是否包含MaxPool2d(2,2)，默认为是
        :return: 卷积块序列
        """
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    # 前向传播函数：数据流过网络的计算流程，必须重写
    def forward(self, x):
        # x是输入图片张量 [batch, 3, 32, 32]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        # 返回预测分数（未经过softmax的logits）
        return x

# 创建模型工具函数，支持加载预训练权重
def create_model(num_classes=10, pretrained_path=None):
    # 实例化网络
    model = CIFAR10CNN(num_classes=num_classes)
    # 如果传入权重路径，加载训练好的模型
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    # 计算模型总参数量，方便查看模型大小
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    return model

# 单独运行此文件测试网络输入输出是否正常
if __name__ == "__main__":
    model = create_model()
    # 模拟1张32×32 RGB图片：[batch=1, 通道3, H32, W32]
    test_img = torch.randn(1, 3, 32, 32)
    output = model(test_img)
    print(f"输入图片shape: {test_img.shape}")
    print(f"模型输出shape: {output.shape}")
