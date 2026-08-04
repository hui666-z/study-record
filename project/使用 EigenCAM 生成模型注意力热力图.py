# 忽略无关警告信息，控制台更干净
import warnings
warnings.filterwarnings('ignore')

# 导入需要的库
import torch        # pytorch深度学习框架
import cv2          # opencv，用来读取、处理图片
import os           # 文件、文件夹操作
import numpy as np  # 数值数组运算
from PIL import Image # 图片保存工具
from ultralytics import YOLO  # ultralytics官方YOLO库
from pytorch_grad_cam import EigenCAM  # CAM热力图算法
from pytorch_grad_cam.utils.image import show_cam_on_image # 热力图叠加原图工具


# =========================================================
# 【重点】模型包装类：解决YOLO原生输出格式和GradCAM不兼容问题
# 原生YOLO推理返回列表对象，CAM库只能接收Tensor张量，必须包装转换
# =========================================================
class ModelWrapper(torch.nn.Module):
    """
    包装YOLO模型，统一模型输出格式
    问题：model(x) 原生返回 Results列表，无法求梯度计算热力图
    作用：拦截输出，提取张量tensor，满足EigenCAM输入要求
    """
    def __init__(self, model):
        super().__init__()  # 继承父类构造函数，固定写法
        self.model = model  # 保存传入的YOLO网络主干

    def forward(self, x):
        """前向传播：输入图片张量x，返回网络预测张量"""
        res = self.model(x)
        # 判断输出是不是列表/元组，YOLO推理默认是列表
        if isinstance(res, (list, tuple)):
            return res[0]  # 取出第0个元素：网络原始预测张量 [1,300,84]
        return res


# =========================================================
# 自定义钩子类：抓取网络中间层特征图
# Hook（钩子）：在网络正向计算的时候，偷偷拿到指定层的输出特征
# pytorch_grad_cam默认钩子和YOLO冲突，所以自己手写实现
# =========================================================
class ActivationsAndGradients:
    def __init__(self, model, target_layers):
        self.model = model                # 传入包装好的YOLO模型
        self.activations = []             # 列表：存放捕获到的特征图
        self.gradients = []               # 本案例EigenCAM暂时不用梯度存储，预留
        self.handles = []                 # 保存钩子句柄，最后用来销毁防止内存泄漏

        # 给每一个目标网络层注册前向钩子
        for target_layer in target_layers:
            hook_handle = target_layer.register_forward_hook(self.save_activation)
            self.handles.append(hook_handle)

    def save_activation(self, module, input, output):
        """
        钩子回调函数：当目标层完成计算自动执行
        module：当前网络层
        input：层的输入
        output：层的输出（特征图！我们需要的东西）
        """
        # .cpu() 放到cpu内存；.detach() 切断梯度计算；存入列表
        self.activations.append(output.cpu().detach())

    def __call__(self, x):
        # 每次推理前清空上次保存的特征图，防止数据残留
        self.activations = []
        return self.model(x) # 执行模型前向推理

    def release(self):
        """程序结束后移除所有钩子，避免持续占用显存/内存"""
        for handle in self.handles:
            handle.remove()


# =========================================================
# YOLO标准预处理函数 letterbox
# YOLO训练和推理统一使用该方式缩放图片，保证图片不会拉伸变形
# =========================================================
def letterbox(im, new_shape=(640, 640)):
    """
    im: opencv读取的原始图片
    new_shape: 网络输入尺寸默认640×640
    return:
        im：缩放+填充后的640×640图片
        r：原图缩放比例
        (dw, dh)：左右、上下灰色填充宽度
    """
    # 获取原图高度、宽度 (H,W)
    shape = im.shape[:2]
    h0, w0 = shape

    # 计算缩放比例：长边等比例缩放，保证完整放进640方框内，不拉伸
    r = min(new_shape[0] / h0, new_shape[1] / w0)

    # 缩放之后、还没填充的图片宽高
    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))

    # 需要填充的总宽度、总高度
    dw_total = new_shape[1] - new_unpad_w
    dh_total = new_shape[0] - new_unpad_h

    # 两边均分填充（居中放置图片）
    dw = dw_total / 2
    dh = dh_total / 2

    # 高质量图片缩放
    im = cv2.resize(im, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    # 上下左右填充灰色(114,114,114)，YOLO官方固定填充色
    top = int(dh)
    bottom = int(new_shape[0] - new_unpad_h - top)
    left = int(dw)
    right = int(new_shape[1] - new_unpad_w - left)
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return im, r, (dw, dh)


# =========================================================
# 主封装类：YOLO热力图可视化工具
# =========================================================
class YOLO26Visualizer:
    def __init__(self, weight, device, layer_idx=18):
        """
        weight：模型权重路径 yolo26n.pt
        device：运行设备 cuda:0 / cpu
        layer_idx：选择可视化哪一层特征
            15 → P3 下采样8倍 小目标特征层
            18 → P4 下采样16倍 中等目标（默认）
            21 → P5 下采样32倍 大目标特征层
        """
        # 创建设备对象
        self.device = torch.device(device)

        # 加载YOLO模型
        self.model_yolo = YOLO(weight)
        # 取出底层神经网络主干(剔除Ultralytics后处理封装)
        raw_model = self.model_yolo.model.to(self.device)

        # 使用上面写的包装类封装网络
        self.wrapped_model = ModelWrapper(raw_model)
        # 设置模型评估模式，关闭dropout、bn训练行为
        self.wrapped_model.eval()

        # 指定我们要观察的网络层
        self.target_layers = [raw_model.model[layer_idx]]

        # 构建EigenCAM热力图工具
        self.cam = EigenCAM(model=self.wrapped_model,
                            target_layers=self.target_layers)

        # 【关键】替换默认特征捕获器，使用我们手写的钩子
        self.cam.activations_and_grads = ActivationsAndGradients(self.wrapped_model, self.target_layers)

    def generate(self, img_path, save_path):
        """
        生成热力图主函数
        img_path：输入图片路径
        save_path：热力图保存路径
        """
        # 1. 读取原始图片 opencv默认格式 BGR
        raw_img = cv2.imread(img_path)

        # 2. YOLO标准预处理：缩放+填充到640×640
        img_640, r, (dw, dh) = letterbox(raw_img)

        # 3. 图片转神经网络需要的张量格式
        # img.shape (H,W,3) → transpose → (3,H,W) 通道在前
        # unsqueeze(0) 增加batch维度 → [1, 3, 640, 640]
        # /255.0 像素值从0~255归一化到0~1
        input_tensor = torch.from_numpy(img_640.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

        # 4. EigenCAM计算灰度热力图 形状[640,640]，数值0~1
        grayscale_cam = self.cam(input_tensor)[0, :]

        # 5. 将热力图叠加到640尺寸图片上
        img_float = np.float32(img_640) / 255.0
        # use_rgb=True 适配PIL保存格式
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # 6. 坐标还原【极易出错！】
        # 裁掉letterbox填充的灰色边框，只保留原图缩放区域
        h_full, w_full = img_640.shape[:2]
        cam_image = cam_image[int(dh): h_full - int(dh), int(dw): w_full - int(dw)]

        # 把图片重新放大回原始图片尺寸
        ori_h, ori_w = raw_img.shape[:2]
        cam_image = cv2.resize(cam_image, (ori_w, ori_h))

        # 7. 保存最终结果
        Image.fromarray(cam_image).save(save_path)
        print(f"✅ 成功！热力图已保存至: {save_path}")


# =========================================================
# 程序入口，代码从此处开始运行
# =========================================================
if __name__ == '__main__':
    # 配置参数
    config = {
        'weight': 'yolo26n.pt',  # 权重文件
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', # 有显卡用GPU，没有自动CPU
        'layer_idx': 18          # 默认可视化P4中等目标层
    }

    # 实例化可视化工具
    visualizer = YOLO26Visualizer(**config)

    # 创建输出文件夹，不存在则新建
    output_dir = 'results_yolo26'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 输入图片路径，改成你自己的图片地址
    img_input = '/gemini/code/ultralytics-main/ultralytics/assets/bus.jpg'
    # 生成并保存热力图
    visualizer.generate(img_input, f'{output_dir}/bus_heatmap.png')