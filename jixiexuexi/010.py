# ==============================================
# 波士顿房价数据集替代方案 - 初学者完整教程
# ==============================================
# 本教程将展示如何：
#   1. 加载并探索数据集
#   2. 数据预处理
#   3. 训练回归模型
#   4. 评估模型性能

# ==============================================
# 1. 导入必要的库
# ==============================================
print("=" * 70)
print("1. 导入必要的库")
print("=" * 70)

# sklearn.datasets: sklearn 内置数据集模块
from sklearn.datasets import make_regression  # 生成回归数据集

# sklearn.model_selection: 模型选择模块（数据划分、交叉验证等）
from sklearn.model_selection import train_test_split  # 划分训练集和测试集

# sklearn.linear_model: 线性模型模块
from sklearn.linear_model import LinearRegression  # 线性回归模型

# sklearn.metrics: 模型评估指标模块
from sklearn.metrics import mean_squared_error, r2_score  # 均方误差、R²分数

# pandas: 数据处理和分析库（类似 Excel）
import pandas as pd  # 用于创建 DataFrame（表格）

# numpy: 数值计算库（数组操作）
import numpy as np  # 用于数学运算

print("\n所有库导入成功！")

# ==============================================
# 2. 生成模拟房价数据集
# ==============================================
print("\n" + "=" * 70)
print("2. 生成模拟房价数据集")
print("=" * 70)

# make_regression(): 生成回归问题的模拟数据
# 原因：California Housing 数据集需要网络下载，可能失败
# 解决方案：使用 make_regression 生成类似的数据

X, y = make_regression(
    n_samples=1000,    # 1000个样本（房屋）
    n_features=8,      # 8个特征（类似 California Housing）
    noise=10.0,        # 添加噪声（模拟真实数据）
    random_state=42    # 固定随机种子（保证每次结果相同）
)

# 参数说明：
#   n_samples: 样本数量（房屋数量）
#   n_features: 特征数量（8个，类似 California Housing）
#   noise: 噪声水平（让数据更接近真实情况）
#   random_state: 随机种子（固定结果）

print(f"\n生成的数据形状:")
print(f"  特征数据 X: {X.shape}")  # (1000, 8)
print(f"  目标变量 y: {y.shape}")  # (1000,)

# ==============================================
# 3. 创建特征名称（模拟 California Housing）
# ==============================================
print("\n" + "-" * 70)
print("3. 创建特征名称（模拟 California Housing）")
print("-" * 70)

# 模拟 California Housing 数据集的8个特征名称
feature_names = [
    'MedInc',      # 收入中位数
    'HouseAge',    # 房屋年龄
    'AveRooms',    # 平均房间数
    'AveBedrms',   # 平均卧室数
    'Population',  # 人口数
    'AveOccup',    # 平均入住人数
    'Latitude',    # 纬度
    'Longitude'    # 经度
]

# 特征含义详解
feature_descriptions = {
    'MedInc': '街区收入中位数（单位：万美元）',
    'HouseAge': '房屋年龄中位数（单位：年）',
    'AveRooms': '平均房间数（每户）',
    'AveBedrms': '平均卧室数（每户）',
    'Population': '街区人口数',
    'AveOccup': '平均入住人数（每户）',
    'Latitude': '纬度（地理位置）',
    'Longitude': '经度（地理位置）'
}

print("\n各特征的详细含义:")
for name in feature_names:
    print(f"  {name:12s}: {feature_descriptions[name]}")#for name in feature_names把name逐一赋值key，feature_descriptions[name]是value
  
print("\n目标变量:")
print("  Price: 房价（单位：十万美元）")

# ==============================================
# 4. 创建 DataFrame（表格形式）
# ==============================================
print("\n" + "=" * 70)
print("4. 创建 DataFrame（表格形式）")
print("=" * 70)

# pd.DataFrame(): 将数组转换为表格格式
df = pd.DataFrame(
    data=X,              # 特征数据
    columns=feature_names  # 列名
)

# 添加目标变量列（房价）
df['Price'] = y  # 新增一列

print(f"\nDataFrame 形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 查看前10行数据
print("\n前10行数据:")
print(df.head(10))

# ==============================================
# 5. 数据统计摘要
# ==============================================
print("\n" + "-" * 70)
print("5. 数据统计摘要")
print("-" * 70)

# describe(): 计算各列的统计量（均值、标准差、最小值、最大值等）
print("\n统计摘要:")
print(df.describe())

# 统计量说明：
#   count: 数据数量
#   mean: 平均值
#   std: 标准差（数据离散程度）
#   min: 最小值
#   25%: 25%分位数（下四分位数）
#   50%: 50%分位数（中位数）
#   75%: 75%分位数（上四分位数）
#   max: 最大值

# ==============================================
# 6. 数据可视化（简单示例）
# ==============================================
print("\n" + "=" * 70)
print("6. 数据可视化示例")
print("=" * 70)

# 查看房价分布
print("\n房价分布统计:")
print(f"  最小值: {df['Price'].min():.2f}")
print(f"  最大值: {df['Price'].max():.2f}")
print(f"  平均值: {df['Price'].mean():.2f}")
print(f"  中位数: {df['Price'].median():.2f}")

# 查看收入与房价的关系
print("\n收入与房价的关系（前5个样本）:")
print(f"  收入中位数: {df['MedInc'].head(5).values}")
print(f"  房价: {df['Price'].head(5).values}")

# ==============================================
# 7. 划分训练集和测试集
# ==============================================
print("\n" + "=" * 70)
print("7. 划分训练集和测试集")
print("=" * 70)

# train_test_split(): 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,              # 特征数据
    y,              # 目标变量
    test_size=0.2,  # 测试集占比 20%
    random_state=42 # 随机种子（保证每次划分结果相同）
)

# 参数说明：
#   test_size=0.2: 测试集占20%，训练集占80%
#   random_state=42: 固定随机种子，确保每次运行结果一致

print(f"\n训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
# 输出: (800, 8) - 800个样本用于训练

print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")
# 输出: (200, 8) - 200个样本用于测试

# ==============================================
# 8. 训练线性回归模型
# ==============================================
print("\n" + "=" * 70)
print("8. 训练线性回归模型")
print("=" * 70)

# LinearRegression(): 创建线性回归模型对象
model = LinearRegression()

# fit(): 训练模型（学习特征与房价的关系）
model.fit(X_train, y_train)

print("\n模型训练完成！")

# 查看模型系数（每个特征对房价的影响）
print("\n模型系数（特征重要性）:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:12s}: {coef:.4f}")
    # 正系数：特征增加 → 房价增加
    # 负系数：特征增加 → 房价降低

print(f"\n截距（intercept）: {model.intercept_:.4f}")

# ==============================================
# 9. 使用模型进行预测
# ==============================================
print("\n" + "=" * 70)
print("9. 使用模型进行预测")
print("=" * 70)

# predict(): 使用训练好的模型预测房价
y_pred = model.predict(X_test)

print(f"\n预测结果形状: {y_pred.shape}")

# 查看前10个预测值 vs 实际值
print("\n前10个样本的预测值 vs 实际值:")
print("  预测值  |  实际值  |  误差")
print("-" * 30)
for i in range(10):
    pred = y_pred[i]
    actual = y_test[i]
    error = pred - actual
    print(f"  {pred:.2f}   |  {actual:.2f}   |  {error:+.2f}")#带正负号

# ==============================================
# 10. 评估模型性能
# ==============================================
print("\n" + "=" * 70)
print("10. 评估模型性能")
print("=" * 70)

# mean_squared_error(): 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)

# r2_score(): 计算 R² 分数（决定系数）
r2 = r2_score(y_test, y_pred)

print("\n模型评估指标:")
print(f"  均方误差 (MSE): {mse:.4f}")
# MSE越小越好，表示预测值与实际值的平均误差

print(f"  R² 分数: {r2:.4f}")
# R²范围：0-1，越接近1越好
# R²=1 表示模型完美预测

# ==============================================
# 11. 实际应用示例
# ==============================================
print("\n" + "=" * 70)
print("11. 实际应用示例 - 预测新房屋价格")
print("=" * 70)

# 假设有一个新房的特征数据
new_house = np.array([
    [5.0,   # 收入中位数：5万美元
     30,    # 房屋年龄：30年
     6.0,   # 平均房间数：6个
     1.0,   # 平均卧室数：1个
     1000,  # 人口：1000人
     3.0,   # 平均入住人数：3人
     37.0,  # 纬度：37.0
     -122.0 # 经度：-122.0
    ]
])

predicted_price = model.predict(new_house)

print("\n新房特征:")
for name, value in zip(feature_names, new_house[0]):
    print(f"  {name:12s}: {value}")

print(f"\n预测房价: {predicted_price[0]:.2f}")

# ==============================================
# 12. 总结
# ==============================================
print("\n" + "=" * 70)
print("12. 完整流程总结")
print("=" * 70)
print("""
机器学习回归任务的标准流程：

  1. 导入库
     ↓
  2. 加载/生成数据集
     ↓
  3. 探索数据（查看形状、特征、统计量）
     ↓
  4. 创建 DataFrame（方便数据处理）
     ↓
  5. 划分数据集（训练集 + 测试集）
     ↓
  6. 训练模型（fit）
     ↓
  7. 预测（predict）
     ↓
  8. 评估（MSE、R²）
     ↓
  9. 应用（预测新数据）

关键概念：
  - X: 特征数据（输入）
  - y: 目标变量（输出）
  - fit(): 训练模型
  - predict(): 预测
  - MSE: 均方误差（越小越好）
  - R²: 决定系数（越接近1越好）

注意：
  - California Housing 数据集需要网络下载
  - 如果下载失败，可以使用 make_regression 生成模拟数据
  - make_regression 的数据结构与真实数据集类似
""")

print("=" * 70)
print("程序执行完成！")
print("=" * 70)
'''1. 明确问题:- 任务类型 ：分类（判断类别）vs 回归（预测数值）,业务目标 ：要解决什么问题？预测什么？,评价指标 ：用什么指标衡量成功？
    ↓
2. 收集数据:- 数据来源 ：数据库、API、文件、爬虫等,数据质量 ：数据是否完整？是否有缺失值？,数据量 ：数据越多越好，但要注意质量
    ↓
3. 数据预处理:- 清洗数据 ：处理缺失值、异常值、重复数据,格式转换 ：文本转数字、日期格式统一,标准化/归一化 ：将数据缩放到合适范围
    ↓
4. 特征工程:- 特征选择 ：选择最有用的特征（去除冗余特征）,特征提取 ：从原始数据中提取新特征,特征转换 ：编码分类变量（独热编码、序数编码）
    ↓
5. 划分数据集:- 训练集 ：用于训练模型（通常占 70-80%）,测试集 ：用于评估模型（通常占 20-30%）
    ↓
6. 选择模型:根据任务类型选择合适的模型：回归任务 ：线性回归、决策树回归、随机森林回归,分类任务 ：逻辑回归、SVM、神经网络
    ↓
7. 训练模型:- 拟合数据 ：让模型学习特征与目标的关系,参数调整 ：优化模型参数（超参数调优）
    ↓
8. 评估模型:- 回归任务 ：均方误差（MSE）、R²分数,分类任务 ：准确率、精确率、召回率、F1分数
    ↓
9. 部署应用:- 保存模型 ：将训练好的模型保存为文件,集成到系统 ：将模型部署到生产环境,监控维护 ：定期更新模型以适应新数据
'''