# ==============================================
# 鸢尾花数据集 - 初学者入门教程（含详细注释）
# ==============================================
# 导入需要的库
from sklearn.datasets import load_iris  # 从sklearn导入鸢尾花数据集
import numpy as np                        # 导入numpy，用于数值计算
import pandas as pd                       # 导入pandas，用于数据处理

# ==============================================
# 1. 加载数据集
# ==============================================
print("=" * 60)  # 打印分隔线（60个等号）
print("1. 加载鸢尾花数据集")  # 打印当前步骤标题
print("=" * 60)  # 打印分隔线

iris = load_iris()  # 加载鸢尾花数据集，返回一个字典-like的对象
# iris 对象包含以下属性：
#   - iris.data: 特征数据（150行，4列）
#   - iris.target: 目标变量（150个0、1、2）
#   - iris.feature_names: 特征名称列表
#   - iris.target_names: 类别名称列表
#   - iris.DESCR: 数据集描述文本

# 打印数据集描述的前100个字符
print(f"\n数据集名称: {iris.DESCR[:100]}...")  
# f"{...}" 是格式化字符串，[:100] 表示取前100个字符

# 打印数据形状
print(f"数据形状: {iris.data.shape}")  
# .shape 返回 (行数, 列数)，这里是 (150, 4) - 150个样本，4个特征

# 打印目标变量形状
print(f"目标变量形状: {iris.target.shape}")  
# 目标变量是一维数组，150个元素

# 打印特征名称
print(f"特征名称: {iris.feature_names}")  
# 输出: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# 打印类别名称
print(f"类别名称: {iris.target_names}")  
# 输出: ['setosa' 'versicolor' 'virginica']

# ==============================================
# 2. 查看原始数据
# ==============================================
print("\n" + "=" * 60)  # 先换行，再打印分隔线
print("2. 查看原始数据")
print("=" * 60)

# 打印前5行特征数据
print("\n前5行特征数据:")
print(iris.data[:5])  # [:5] 表示取前5个元素（切片操作）
# iris.data 是一个二维数组，每行代表一个样本

# 打印前5行目标变量
print("\n前5行目标变量:")
print(iris.target[:5])  # 前5个样本的类别标签

# 打印目标变量对应的类别
print("\n目标变量对应的类别:")
# enumerate() 函数同时返回索引和值
for i, name in enumerate(iris.target_names):
    print(f"  {i} = {name}")  # i是0、1、2，name是对应的类别名

# ==============================================
# 3. 基础统计计算
# ==============================================
print("\n" + "=" * 60)
print("3. 基础统计计算")
print("=" * 60)

# 循环计算每个特征的统计量
for i, feature_name in enumerate(iris.feature_names):
    # i是特征索引（0-3），feature_name是特征名称
    feature_data = iris.data[:, i]  # 取第i列的所有数据
    # : 表示所有行，i 表示第i列
    
    print(f"\n特征: {feature_name}")
    # np.min(): 计算最小值
    print(f"  最小值: {np.min(feature_data):.2f}")
    # :.2f 表示保留2位小数
    
    # np.max(): 计算最大值
    print(f"  最大值: {np.max(feature_data):.2f}")
    
    # np.mean(): 计算平均值（均值）
    print(f"  平均值: {np.mean(feature_data):.2f}")
    
    # np.median(): 计算中位数
    print(f"  中位数: {np.median(feature_data):.2f}")
    
    # np.std(): 计算标准差（Standard Deviation）
    print(f"  标准差: {np.std(feature_data):.2f}")
    
    # np.var(): 计算方差（Variance）
    print(f"  方差: {np.var(feature_data):.2f}")

# ==============================================
# 4. 按类别分组统计
# ==============================================
print("\n" + "=" * 60)
print("4. 按类别分组统计")
print("=" * 60)

# 转换为 DataFrame 方便处理
df = pd.DataFrame(
    data=iris.data,  # 数据内容
    columns=iris.feature_names  # 列名
)
# DataFrame 是 pandas 的表格数据结构，类似 Excel 表格

# 添加 species（物种）列
# 列表推导式：把 0、1、2 转换为对应的类别名称
df['species'] = [iris.target_names[i] for i in iris.target]
# 等价于：
# species_list = []
# for i in iris.target:
#     species_list.append(iris.target_names[i])
# df['species'] = species_list

# 打印前10行数据
print("\n前10行数据:")
print(df.head(10))  # .head(n) 显示前n行，默认是5行

# 打印每个类别的样本数
print("\n每个类别的样本数:")
print(df['species'].value_counts())  # .value_counts() 统计每个值出现的次数

# 按类别统计平均值
print("\n按类别统计的平均值:")
print(df.groupby('species').mean())
# .groupby('species') 按物种分组
# .mean() 计算每组的平均值

# ==============================================
# 5. 相关性分析
# ==============================================
print("\n" + "=" * 60)
print("5. 相关性分析")
print("=" * 60)

# 计算特征之间的相关系数矩阵
# df.drop('species', axis=1) 删除 species 列
# .corr() 计算相关系数矩阵
corr_matrix = df.drop('species', axis=1).corr()

print("\n特征相关系数矩阵:")
print(corr_matrix)
# 相关系数范围 [-1, 1]
# 1: 完全正相关
# 0: 不相关
# -1: 完全负相关

# 找出相关性最强的特征对
print("\n相关性最强的特征对:")
correlation_pairs = []  # 空列表，用于存储结果

# 双重循环：遍历所有特征对
for i in range(len(iris.feature_names)):  # i从0到3
    for j in range(i+1, len(iris.feature_names)):  # j从i+1到3，避免重复
        corr = corr_matrix.iloc[i, j]  # 取第i行第j列的相关系数
        # .iloc[i,j] 是按位置索引
        correlation_pairs.append((
            abs(corr),  # 绝对值（用于排序）
            iris.feature_names[i],  # 特征1名称
            iris.feature_names[j],  # 特征2名称
            corr  # 原始相关系数
        ))

# 按绝对值从大到小排序
correlation_pairs.sort(reverse=True, key=lambda x: x[0])
# sort(): 排序
# reverse=True: 降序（从大到小）
# key=lambda x: x[0]: 按元组的第0个元素（绝对值）排序

# 打印前3对相关性最强的特征
for corr_abs, f1, f2, corr in correlation_pairs[:3]:
    print(f"  {f1} 与 {f2}: 相关系数 = {corr:.4f}")

# ==============================================
# 6. 简单的预测示例
# ==============================================
print("\n" + "=" * 60)
print("6. 简单的分类预测 - 基于特征阈值")
print("=" * 60)

# 选择一个特征进行简单分类（使用花瓣宽度）
# .index() 查找列表中元素的位置
petal_width_col = iris.feature_names.index('petal width (cm)')
# 返回 3，因为 'petal width (cm)' 在第3个位置（索引从0开始）

# 提取花瓣宽度这一列的所有数据
petal_widths = iris.data[:, petal_width_col]
# : 表示所有行，petal_width_col（即3）表示第3列

# 计算每个类别的平均花瓣宽度
avg_widths = {}  # 空字典，用于存储结果

for i, name in enumerate(iris.target_names):
    # iris.target == i: 返回布尔数组，只保留类别为i的样本
    # petal_widths[...] 只取对应位置的值
    avg_width = np.mean(petal_widths[iris.target == i])
    avg_widths[name] = avg_width  # 存入字典
    print(f"{name} 平均花瓣宽度: {avg_width:.2f} cm")

# 使用简单的规则进行预测
print("\n简单分类规则:")
print("- 花瓣宽度 < 0.8: setosa")
print("- 0.8 ≤ 花瓣宽度 < 1.8: versicolor")
print("- 花瓣宽度 ≥ 1.8: virginica")

# 测试预测
print("\n预测结果:")
correct = 0  # 正确预测计数器，初始为0

for i in range(len(iris.data)):  # 遍历所有150个样本
    # 获取当前样本的花瓣宽度
    width = iris.data[i, petal_width_col]
    
    # 获取真实标签
    true_label = iris.target_names[iris.target[i]]
    
    # 简单的预测规则（基于阈值）
    if width < 0.8:  # 如果小于0.8
        pred_label = 'setosa'
    elif width < 1.8:  # 如果在0.8到1.8之间
        pred_label = 'versicolor'
    else:  # 如果大于等于1.8
        pred_label = 'virginica'
    
    # 如果预测正确，计数器加1
    if pred_label == true_label:
        correct += 1

# 计算准确率
accuracy = correct / len(iris.data) * 100  # 转换为百分比
print(f"\n正确预测数: {correct}/{len(iris.data)}")
print(f"准确率: {accuracy:.2f}%")

# ==============================================
# 7. 保存数据到文件
# ==============================================
print("\n" + "=" * 60)
print("7. 保存数据到文件")
print("=" * 60)

# 保存为 CSV 文件
csv_path = "d:/coding/test.p/phthon/iris_data.csv"  # 文件路径
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
# .to_csv(): 保存为CSV文件
# index=False: 不保存行索引（0、1、2...）
# encoding='utf-8-sig': 使用支持中文的编码

print(f"\n数据已保存到: {csv_path}")

# ==============================================
# 使用条件和注意事项
# ==============================================
print("\n" + "=" * 60)
print("使用条件和注意事项")
print("=" * 60)
print("""
1. 适合初学者: 鸢尾花数据集是经典的入门数据集
2. 数据量小: 只有150个样本，计算速度快
3. 特征简单: 只有4个特征，容易理解
4. 类别均衡: 每个类别正好50个样本
数据要为正
5. 适用场景:
   - 分类算法学习
   - 数据可视化练习
   - 特征工程入门
   - 模型评估基础
""")
