# ==============================================
# 机器学习实验作业
# ==============================================
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold

# ==============================================
# 实验一：数据规范化
# ==============================================
print("=" * 70)
print("实验一：数据规范化")
print("=" * 70)

# 给定学生成绩数据
# 每行一个学生，两列分别是数学和英语成绩
scores = np.array([
    [85, 520],   # 学生1: 数学85分，英语520分
    [90, 580],   # 学生2: 数学90分，英语580分
    [78, 490],   # 学生3: 数学78分，英语490分
    [92, 610],   # 学生4: 数学92分，英语610分
    [88, 550]    # 学生5: 数学88分，英语550分
])

print("\n原始成绩数据:")
print(scores)
print("  第0列: 数学成绩")
print("  第1列: 英语成绩")

# 1. Min-Max 归一化
print("\n" + "-" * 70)
print("1. Min-Max 归一化（映射到 [0,1] 区间")
print("-" * 70)

minmax_scaler = MinMaxScaler()  # 创建归一化工具
scores_minmax = minmax_scaler.fit_transform(scores)  # 学习并转换
print("\nMin-Max 归一化结果:")
print(scores_minmax)
print("\n最小值最大值:")
print(f"  数学: min={scores[:, 0].min()}, max={scores[:, 0].max()}")
print(f"  英语: min={scores[:, 1].min()}, max={scores[:, 1].max()}")
print("\nMin-Max 公式: (x - min) / (max - min)")

# 2. Z-score 标准化
print("\n" + "-" * 70)
print("2. Z-score 标准化（均值为0，标准差为1）")
print("-" * 70)

standard_scaler = StandardScaler()  # 创建标准化工具
scores_standard = standard_scaler.fit_transform(scores)  # 学习并转换
print("\nZ-score 标准化结果:")
print(scores_standard)
print("\n统计量:")
print(f"  数学: 均值={scores[:, 0].mean():.2f}, 标准差={scores[:, 0].std():.2f}")
print(f"  英语: 均值={scores[:, 1].mean():.2f}, 标准差={scores[:, 1].std():.2f}")
print("\nZ-score 公式: (x - mean) / std")

# 两种方法的区别
print("\n" + "-" * 70)
print("两种方法的区别:")
print("-" * 70)
print("Min-Max 归一化:")
print("  [√] 输出范围固定: [0,1]")
print("  [√] 保留数据相对关系")
print("  [√] 适合需要固定范围的算法（如神经网络）")
print("  [×] 对异常值敏感")
print()
print("Z-score 标准化:")
print("  [√] 输出范围不固定，均值为0，标准差为1")
print("  [√] 适合大多数算法（如SVM、线性回归）")
print("  [√] 对异常值较不敏感")
print("  [×] 不会限制范围")


# ==============================================
# 实验二：特征编码
# ==============================================
print("\n" + "=" * 70)
print("实验二：特征编码")
print("=" * 70)

# 学生信息数据
# 列1: 专业（类别：计算机、数学、物理）
# 列2: 学历（有序：本科、硕士、博士）
student_data = np.array([
    ["计算机", "本科"],
    ["数学", "硕士"],
    ["计算机", "博士"],
    ["物理", "本科"]
])

print("\n原始学生信息:")
print(student_data)

# 1. 对"专业"进行独热编码
print("\n" + "-" * 70)
print("1. 对'专业'进行独热编码")
print("-" * 70)

major = student_data[:, 0].reshape(-1, 1)  # 提取专业列
'''student_data = [
    ["计算机", "本科"],  # 第0行
    ["数学", "硕士"],    # 第1行
    ["计算机", "博士"],   # 第2行
    ["物理", "本科"]     # 第3行
]
    ↓
student_data[:, 0]  # 提取第0列
    ↓
['计算机', '数学', '计算机', '物理']
    ↓
.reshape(-1, 1)  # 变成4行1列
    ↓
[['计算机']      # 第0行
 ['数学']        # 第1行
 ['计算机']      # 第2行
 ['物理']]       # 第3行
    ↓
用于 OneHotEncoder 编码'''
onehot = OneHotEncoder(sparse_output=False)  # 创建独热编码器,sparse_output:稀疏矩阵 ：只存储非零位置的值，节省内存
major_encoded = onehot.fit_transform(major)  # 学习并转换
print("\n专业独热编码结果:")
print(major_encoded)
print(f"\n专业类别映射:")
for i, category in enumerate(onehot.categories_[0]):
    onehot_vec = np.zeros(len(onehot.categories_[0]))
    onehot_vec[i] = 1
    print(f"  {category} -> {onehot_vec}")

# 2. 对"学历"进行序数编码
print("\n" + "-" * 70)
print("2. 对'学历'进行序数编码")
print("-" * 70)

education = student_data[:, 1].reshape(-1, 1)  # 提取学历列
ordinal = OrdinalEncoder(categories=[["本科", "硕士", "博士"]])  # 指定顺序：本科=0, 硕士=1, 博士=2
education_encoded = ordinal.fit_transform(education)  # 学习并转换
print("\n学历序数编码结果:")
print(education_encoded)
print(f"\n学历类别映射:")
for i, category in enumerate(["本科", "硕士", "博士"]):
    print(f"  {category} -> {i}")

# 合并所有编码后的数据
print("\n" + "-" * 70)
print("3. 合并编码后的完整矩阵:")
print("-" * 70)
encoded_all = np.hstack([major_encoded, education_encoded])
print(encoded_all)


# ==============================================
# 实验三：方差过滤法
# ==============================================
print("\n" + "=" * 70)
print("实验三：方差过滤法")
print("=" * 70)

# 数据
# 注意：第3列全是5，方差为0
X = np.array([
    [1, 10, 5, 100],
    [2, 20, 5, 200],
    [1, 30, 5, 300],
    [2, 40, 5, 400],
    [1, 50, 5, 500],
    [2, 60, 5, 600]
])

print("\n原始数据:")
print(X)

# 先计算每个特征的方差
print("\n" + "-" * 70)
print("计算各特征的方差:")
print("-" * 70)

for i in range(X.shape[1]):
    '''# shape 返回 (行数, 列数)
X.shape  # (6, 4) - 6个样本，4个特征
X.shape[0]  # 6 - 样本数
X.shape[1]  # 4 - 特征数'''
    variance = np.var(X[:, i])
    print(f"特征{i}: 方差={variance:.2f}")
    '''# np.var() 计算方差
# 方差公式：var = mean((x - mean)²)
# 表示数据的离散程度

np.var([1, 2, 1, 2, 1, 2])  # 特征0的方差
np.var([10, 20, 30, 40, 50, 60])  # 特征1的方差
np.var([5, 5, 5, 5, 5, 5])  # 特征2的方差
np.var([100, 200, 300, 400, 500, 600])  # 特征3的方差'''

# 1. 使用 VarianceThreshold 过滤方差<1的特征
print("\n" + "-" * 70)
print("1. 使用 VarianceThreshold 过滤方差<1的特征")
print("-" * 70)

selector = VarianceThreshold(threshold=1)  # 方差阈值设为1
X_filtered = selector.fit_transform(X)  # 学习并转换
print("\n保留的特征索引:")
print(selector.get_support(indices=True))

print("\n过滤后的数据:")
print(X_filtered)

# 说明被过滤的特征
print("\n" + "-" * 70)
print("3. 被过滤的特征:")
print("-" * 70)

all_indices = list(range(X.shape[1]))
selected_indices = selector.get_support(indices=True)
filtered_indices = [i for i in all_indices if i not in selected_indices]
print(f"被过滤的特征: {filtered_indices}")
for i in filtered_indices:
    print(f"  特征{i}: 方差={np.var(X[:, i]):.2f} < 1")
print("\n原因:")
print("  方差小意味着所有样本的值几乎一样，几乎没有信息")
print("  这样的特征对机器学习模型没有帮助，可以删除")

print("\n" + "=" * 70)
print("实验完成！")
print("=" * 70)
