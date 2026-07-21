# ==============================================
# 递归特征消除 (RFE) - 特征选择方法
# ==============================================
# 导入需要的库
from sklearn.feature_selection import RFE  # 递归特征消除
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn import datasets  # sklearn 内置数据集

# ==============================================
# 1. 加载鸢尾花数据集
# ==============================================
print("=" * 70)
print("1. 加载鸢尾花数据集")
print("=" * 70)

iris = datasets.load_iris()  # 加载鸢尾花数据集

# iris 包含：
#   - data: 特征数据 (150个样本, 4个特征)
#   - target: 目标变量 (150个类别标签)
#   - feature_names: 特征名称

X = iris.data  # 特征矩阵：150行 x 4列
y = iris.target  # 目标变量：150个标签
feature_names = iris.feature_names  # 特征名称列表

print(f"\n特征数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"特征名称: {feature_names}")
# 输出: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# ==============================================
# 2. 创建 RFE 特征选择器
# ==============================================
print("\n" + "=" * 70)
print("2. 创建 RFE 特征选择器")
print("=" * 70)

# RFE: Recursive Feature Elimination（递归特征消除）
# 原理：反复构建模型，每次移除最不重要的特征
selector = RFE(
    estimator=LogisticRegression(),  # 基学习器：逻辑回归
    n_features_to_select=2,          # 最终保留2个特征
    step=1                           # 每次移除1个特征
)
# 参数说明：
#   estimator: 用于评估特征重要性的模型
#   n_features_to_select: 要保留的特征数量
#   step: 每次迭代移除的特征数量

# ==============================================
# 3. 训练模型并进行特征选择
# ==============================================
print("\n" + "=" * 70)
print("3. 训练模型并进行特征选择")
print("=" * 70)

selector.fit(X, y)  # 训练模型并选择特征
# fit() 过程：
#   1. 用所有特征训练逻辑回归
#   2. 评估每个特征的重要性
#   3. 移除最不重要的1个特征
#   4. 重复直到只剩2个特征

# ==============================================
# 4. 查看特征选择结果
# ==============================================
print("\n" + "=" * 70)
print("4. 查看特征选择结果")
print("=" * 70)

# selector.support_: 布尔数组，表示每个特征是否被选中
# True = 被选中，False = 被移除
print("\n特征选择状态:")
for name, selected in zip(feature_names, selector.support_):
    # zip() 将两个列表合并成元组列表
    # 例如：[('sepal length', True), ('sepal width', False), ...]
    
    status = '选中' if selected else '未选中'
    # if selected 为 True，status = '选中'
    # if selected 为 False，status = '未选中'
    
    print(f"  {name:20s}: {status}")
    # {name:20s} 表示左对齐的20字符字符串

# ==============================================
# 5. 查看特征排名
# ==============================================
print("\n" + "-" * 70)
print("5. 查看特征排名")
print("-" * 70)

# selector.ranking_: 每个特征的排名
# 排名规则：1=最终保留，其他数字=被移除的顺序
# 例如：[2, 4, 1, 3] 表示：
#   特征2排名第1（最重要，被保留）
#   特征4排名第2
#   特征1排名第3
#   特征3排名第4（最不重要）

print("\n特征排名（1=最重要，最终保留）:")
for name, rank in zip(feature_names, selector.ranking_):
    print(f"  {name:20s}: 排名 {rank}")

# 按重要性排序（从高到低）
print("\n按重要性排序（从高到低）:")
feature_importance = sorted(zip(selector.ranking_, feature_names), reverse=True)
# sorted() 排序，默认按元组第一个元素排序
# reverse=True 表示降序排列
# zip() 合并排名和名称

for rank, name in feature_importance:
    importance = "最重要" if rank == 1 else "较重要"
    print(f"  {name:20s}: {importance}")

# ==============================================
# 6. 转换数据（保留选中的特征）
# ==============================================
print("\n" + "=" * 70)
print("6. 转换数据（保留选中的特征）")
print("=" * 70)

X_new = selector.transform(X)
# transform() 只保留被选中的特征

print(f"\n原始数据形状: {X.shape}")
print(f"选择后数据形状: {X_new.shape}")
# 预期输出：(150, 2) - 从4列减少到2列

# ==============================================
# 7. 获取选中特征的名称
# ==============================================
print("\n" + "-" * 70)
print("7. 获取选中特征的名称")
print("-" * 70)

# get_support(indices=True) 返回选中特征的索引
selected_indices = selector.get_support(indices=True)
# 例如：[2, 3] 表示第2和第3个特征被选中

selected_features = [feature_names[i] for i in selected_indices]
# 列表推导式：从索引获取特征名称

print(f"\n被选中的特征（索引 {list(selected_indices)}）:")
for feature in selected_features:
    print(f"  - {feature}")

# ==============================================
# 8. 完整流程总结
# ==============================================
print("\n" + "=" * 70)
print("8. RFE 算法流程总结")
print("=" * 70)
print("""
RFE（递归特征消除）算法步骤：
  1. 用所有特征训练基学习器（逻辑回归）
  2. 根据模型系数计算特征重要性
  3. 移除最不重要的1个特征（step=1）
  4. 用剩余特征重新训练模型
  5. 重复步骤2-4，直到只剩 n_features_to_select 个特征

本例中：
  - 原始特征：4个（sepal length, sepal width, petal length, petal width）
  - 目标保留：2个
  - 最终结果：保留最重要的2个特征
""")

print("=" * 70)
print("程序执行完成！")
print("=" * 70)
