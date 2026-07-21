# ==============================================
# 方差阈值特征选择 - 优化版（含详细注释）
# ==============================================
# 导入需要的库
from sklearn.feature_selection import VarianceThreshold  # 从sklearn导入方差阈值选择器
import numpy as np                                        # 导入numpy，用于数值计算

# 示例数据（5个样本，3个特征）
# 每个内部列表代表一个样本，包含3个特征值
data = [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]]
# 数据说明：
# 样本0: [0, 0, 1]
# 样本1: [0, 1, 0]
# 样本2: [0, 0, 1]
# 样本3: [1, 0, 0]
# 样本4: [1, 1, 1]

print("原始数据:")
print(np.array(data))  # 转换为numpy数组并打印，方便查看
print(f"数据形状: {np.array(data).shape}")  # .shape 返回 (5, 3) - 5个样本，3个特征

# ==============================================
# 方法1: 使用默认阈值（threshold=0，移除方差为0的特征）
# ==============================================
print("\n" + "="*50)  # 换行后打印分隔线（50个等号）
print("方法1: 默认阈值 (threshold=0)")
print("="*50)

# 创建方差阈值选择器
selector = VarianceThreshold()  # 不指定参数，默认 threshold=0
# threshold=0 的意思是：只移除方差为0的特征（即所有值都相同的特征）

try:  # try-except 异常处理，防止报错
    # fit_transform() = fit() + transform()
    # fit(): 学习数据，计算每个特征的方差
    # transform(): 根据方差阈值筛选特征
    y = selector.fit_transform(data)
    
    # selector.variances_ 是自动计算得到的每个特征的方差
    # 注意：下划线 _ 表示这是通过 fit() 学习得到的属性
    print(f"\n各特征方差: {selector.variances_}")
    
    # 打印筛选后的数据形状
    print(f"\n选择后的特征形状: {y.shape}")
    print("选择后的特征:")
    print(y)
    
    # get_support(indices=True) 返回被选中的特征的索引
    # indices=True 表示返回索引，而不是布尔数组
    selected_indices = selector.get_support(indices=True)
    print(f"\n被选中的特征索引: {selected_indices}")
    
except ValueError as e:  # 捕获可能的异常
    print(f"错误: {e}")

# ==============================================
# 方法2: 使用较低的阈值（根据数据特点调整）
# ==============================================
print("\n" + "="*50)
print("方法2: 较低阈值 (threshold=0.1)")
print("="*50)

# 创建方差阈值选择器，设置 threshold=0.1
selector2 = VarianceThreshold(threshold=0.1)
# threshold=0.1 的意思是：只保留方差大于0.1的特征

try:
    y2 = selector2.fit_transform(data)
    
    print(f"\n各特征方差: {selector2.variances_}")
    print(f"\n选择后的特征形状: {y2.shape}")
    print("选择后的特征:")
    print(y2)
    
    selected_indices2 = selector2.get_support(indices=True)
    print(f"\n被选中的特征索引: {selected_indices2}")
    
except ValueError as e:
    print(f"错误: {e}")

# ==============================================
# 方法3: 手动计算方差（帮助理解原理）
# ==============================================
print("\n" + "="*50)
print("方法3: 手动计算方差（原理说明）")
print("="*50)

# 把数据转换为numpy数组
data_np = np.array(data)

# 循环遍历每个特征（每一列）
for i in range(data_np.shape[1]):  # shape[1] 是列数，即特征数（3）
    # 提取第 i 列的所有数据
    feature = data_np[:, i]  # : 表示所有行，i 表示第 i 列
    
    # 计算这个特征的方差
    variance = np.var(feature)  # np.var() 计算方差
    
    # 打印结果，保留4位小数
    print(f"特征{i} 的方差: {variance:.4f}")
    # 方差的计算公式：var = mean((x - mean)^2)
    # 即：先求平均值，再求每个值与平均值差的平方的平均值


# ==============================================
# 皮尔逊相关系数特征选择
# ==============================================
print("\n" + "="*50)
print("皮尔逊相关系数特征选择")
print("="*50)

# 导入需要的库
from sklearn.datasets import make_regression  # 用于生成人工回归数据
from scipy.stats import pearsonr             # 计算皮尔逊相关系数

# pearsonr 返回一个元组 (相关系数, p值)
# 相关系数 r 范围 [-1, 1]
#   -1: 完全负相关
#    0: 不相关
#    1: 完全正相关
# p值用于判断相关性是否显著

# 生成回归数据（100个样本，3个特征）
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
# n_samples: 样本数量
# n_features: 特征数量
# noise: 添加的噪声大小，让数据更真实
# random_state: 随机种子，确保每次运行结果一致
# X: 特征数据 (100, 3)
# y: 目标变量 (100,)

print(f"\n数据形状: X={X.shape}, y={y.shape}")

# 计算每个特征与目标变量的相关系数
print("\n各特征与目标变量的皮尔逊相关系数:")
for i in range(X.shape[1]):  # 遍历每个特征
    # 计算第 i 个特征与 y 的相关系数
    corr, p_value = pearsonr(X[:, i], y)
    # X[:, i]: 第 i 列的所有数据
    # pearsonr(x, y): 计算 x 和 y 的相关系数
    # corr: 相关系数
    # p_value: p值（统计显著性）
    
    # 打印结果
    print(f"特征{i}: 相关系数={corr:.4f}, p值={p_value:.4e}")
    # .4f: 保留4位小数
    # .4e: 用科学计数法保留4位小数

# 找出相关性最强的特征
correlations = []  # 空列表，存储每个特征的相关系数绝对值
for i in range(X.shape[1]):
    corr, _ = pearsonr(X[:, i], y)
    #       ↑
    #       下划线 _ 是约定俗成的"丢弃变量"
    #       表示我们不需要这个返回值（p值）
    correlations.append(abs(corr))  # 取绝对值并添加到列表

# np.argmax() 返回最大值的索引
best_feature = np.argmax(correlations)
print(f"\n相关性最强的特征: 特征{best_feature} (|r|={correlations[best_feature]:.4f})")

# 下面是多行注释，解释索引语法
'''X[:, i]  
#   ↑  ↑
#   │  └── 第 i 列（第 i 个特征）
#   └───── 所有行（所有样本）'''
