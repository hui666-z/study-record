# ==============================================
# 数据预处理示例 - 初学者教程（含详细注释）
# ==============================================
# 先准备环境，在终端里执行以下命令：
# conda activate sklearn    # 激活 sklearn 环境
# python d:\coding\test.p\python\06.py  # 运行本脚本

# 1. 导入需要的库
# StandardScaler: 用于数据标准化（均值为0，标准差为1）
from sklearn.preprocessing import StandardScaler  # 导入标准化工具
# MinMaxScaler: 用于数据归一化（缩放到指定范围，默认0-1）
from sklearn.preprocessing import MinMaxScaler    # 导入归一化工具
# OneHotEncoder: 用于独热编码（处理分类数据）
from sklearn.preprocessing import OneHotEncoder   # 导入独热编码工具
# numpy: 用于数值计算和数组操作
import numpy as np  # 导入 numpy 并简写为 np


# ==============================================
# 示例1: 数据标准化 (Standardization)
# ==============================================
# 标准化会将数据转换为均值为0，标准差为1的分布
# 公式: (x - mean) / std
#   - x: 原始数据
#   - mean: 该特征的平均值
#   - std: 该特征的标准差
# 为什么用标准化？让不同量纲的特征有可比性

# 创建一个3行3列的示例数据
# 每行是一个样本，每列是一个特征
X = np.array([[1, 2, 3],  # 样本0: 特征0=1, 特征1=2, 特征2=3
              [4, 5, 6],  # 样本1: 特征0=4, 特征1=5, 特征2=6
              [7, 8, 9]]) # 样本2: 特征0=7, 特征1=8, 特征2=9

print("原始数据:")
print(X)  # 打印原始数据

# 创建标准化器对象
scaler = StandardScaler()  # () 表示实例化，创建一个工具对象

# 拟合数据并转换（fit计算均值和标准差，transform进行转换）
# fit_transform() = fit() + transform()
#   fit(): 学习数据，计算每列的均值和标准差
#   transform(): 用学习到的参数转换数据
X_scaled = scaler.fit_transform(X)

print("\n标准化后的数据:")
print(X_scaled)  # 打印标准化后的数据
# 标准化后的特点: 每列的均值≈0，标准差≈1
# 例如第一列 [1,4,7] 的均值是4，标准差是3
# 标准化后：(1-4)/3=-1, (4-4)/3=0, (7-4)/3=1 → [-1, 0, 1]


# ==============================================
# 示例2: 数据归一化 (Normalization / Min-Max Scaling)
# ==============================================
# 归一化会将数据缩放到指定范围（默认0到1）
# 公式: (x - min) / (max - min)
#   - x: 原始数据
#   - min: 该特征的最小值
#   - max: 该特征的最大值
# 为什么用归一化？当需要把数据压缩到固定区间时使用

# 创建归一化器对象
scaler = MinMaxScaler()  # 这里变量名仍用 scaler，覆盖了之前的标准化器

# 拟合数据并转换
# fit() 学习每列的最小值和最大值
# transform() 用学习到的参数转换数据
X_scaled = scaler.fit_transform(X)

print("\n归一化后的数据:")
print(X_scaled)
# 归一化后的特点: 所有值都在0到1之间
# 例如第一列 [1,4,7]，min=1, max=7
# 归一化后：(1-1)/6=0, (4-1)/6=0.5, (7-1)/6=1 → [0, 0.5, 1]


# ==============================================
# 示例3: 独热编码 (One-Hot Encoding)
# ==============================================
# 独热编码用于处理分类数据（非数值型数据，如类别、标签）
# 将类别转换为二进制向量，每个类别对应一个维度
# 为什么用独热编码？机器学习算法只能处理数值型数据

# 创建分类数据（城市和性别）
# 每行是一个样本，有两列：城市、性别
data = np.array([['BJ', 'X'],  # 样本0: 城市=BJ, 性别=X
                 ['SH', 'Y'],  # 样本1: 城市=SH, 性别=Y
                 ['SZ', 'Y']]) # 样本2: 城市=SZ, 性别=Y

print("\n原始分类数据:")
print(data)  # 打印原始分类数据

# 创建独热编码器对象
enc = OneHotEncoder()  # enc 是 encoder 的简写

# 拟合数据（学习类别）
# fit() 会找出每列有多少种不同的值
# 例如第一列有 BJ、SH、SZ 三种，第二列有 X、Y 两种
enc.fit(data)

# 进行编码转换
# transform() 会将每个类别转换为二进制向量
data_encoded = enc.transform(data)
# 注意：transform() 返回的是稀疏矩阵（只存储非零元素，节省内存）

print("\n独热编码后的数据:")
print(data_encoded.toarray())  # 转换为普通数组才能打印
# toarray(): 将稀疏矩阵转换为密集（普通）数组
# 独热编码后的特点: 
# - BJ→[1,0,0], SH→[0,1,0], SZ→[0,0,1] （城市有3种，所以3个维度）
# - X→[1,0], Y→[0,1] （性别有2种，所以2个维度）
# 合并后：
# BJ,X → [1,0,0,1,0]
# SH,Y → [0,1,0,0,1]
# SZ,Y → [0,0,1,0,1]

'''
# 这两行代码的效果是一样的：
data_encoded = enc.fit_transform(data)        # 方法1：一步到位（推荐）

# 等价于
enc.fit(data)                                  # 先学习
data_encoded = enc.transform(data)             # 后转换
'''

'''
# sklearn 的标准工作流程（通用模式）：

# 1. 创建工具（实例化）
工具 = SomeClass()  # SomeClass 是包里的类，如 StandardScaler、MinMaxScaler

# 2. 训练工具（学习数据特征）
工具.fit(数据)  # fit() 方法：从数据中学习参数

# 3. 使用工具（转换数据）
新数据 = 工具.transform(数据)  # transform() 方法：应用学习到的参数

# 或者两步合一（更方便）
新数据 = 工具.fit_transform(数据)  # fit() + transform() 合并执行
'''
