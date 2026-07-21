# ==============================================
# 四种经典机器学习算法综合示例
# ==============================================
# 包含：线性回归、逻辑回归、决策树、K最近邻
# 数据集：鸢尾花数据集（分类任务）和波士顿房价数据集（回归任务）

# ==============================================
# 1. 导入必要的库
# ==============================================
print("=" * 70)
print("1. 导入必要的库")
print("=" * 70)

# ========== 数据集模块 ==========
# sklearn.datasets: 提供内置数据集和数据生成工具
# load_iris: 加载鸢尾花数据集（经典分类数据集，包含150个样本，4个特征）
# make_regression: 生成模拟回归数据（避免下载真实数据集的权限问题）
from sklearn.datasets import load_iris, make_regression

# ========== 模型模块 ==========
# sklearn.linear_model: 线性模型模块
# LinearRegression: 线性回归模型（用于回归任务，预测连续值）
# LogisticRegression: 逻辑回归模型（用于分类任务，判断类别）
from sklearn.linear_model import LinearRegression, LogisticRegression

# sklearn.tree: 决策树模块
# DecisionTreeClassifier: 决策树分类器（用于分类任务）
# DecisionTreeRegressor: 决策树回归器（用于回归任务）
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# sklearn.neighbors: 近邻算法模块
# KNeighborsClassifier: K近邻分类器（基于距离的分类算法）
# KNeighborsRegressor: K近邻回归器（基于距离的回归算法）
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# ========== 工具模块 ==========
# sklearn.model_selection: 模型选择模块
# train_test_split: 将数据集划分为训练集和测试集（评估模型泛化能力）
from sklearn.model_selection import train_test_split

# ========== 评估指标模块 ==========
# sklearn.metrics: 评估指标模块
# accuracy_score: 计算准确率（分类任务评估指标）
# classification_report: 生成详细分类报告（包含精确率、召回率、F1分数）
# mean_squared_error: 计算均方误差（回归任务评估指标，值越小越好）
# r2_score: 计算R²分数（回归任务评估指标，范围0-1，越接近1越好）
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
)

# ========== 基础库 ==========
# numpy (np): 高性能数值计算库，用于数组操作和矩阵运算
import numpy as np

# pandas (pd): 数据分析库，提供DataFrame数据结构，用于数据处理和分析
import pandas as pd

print("\n所有库导入成功！")

# ==============================================
# 2. 分类任务示例（鸢尾花数据集）
# ==============================================
print("\n" + "=" * 70)
print("2. 分类任务示例 - 鸢尾花数据集")
print("=" * 70)

# 加载数据集
iris = load_iris()
X_class = iris.data    # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y_class = iris.target  # 目标：0=山鸢尾, 1=变色鸢尾, 2=维吉尼亚鸢尾

print(f"\n数据集信息:")
print(f"  特征数据形状: {X_class.shape}")
print(f"  目标变量形状: {y_class.shape}")
print(f"  类别名称: {iris.target_names}")

# 划分训练集和测试集
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

print(f"\n训练集: {X_train_c.shape[0]}个样本")
print(f"测试集: {X_test_c.shape[0]}个样本")

# ==============================================
# 3. 逻辑回归分类
# ==============================================
print("\n" + "-" * 70)
print("3. 逻辑回归分类")
print("-" * 70)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_c, y_train_c)
y_pred_log = log_reg.predict(X_test_c)

acc_log = accuracy_score(y_test_c, y_pred_log)
print(f"\n逻辑回归准确率: {acc_log:.4f}")
print("\n分类报告:")
print(classification_report(y_test_c, y_pred_log, target_names=iris.target_names))

# ==============================================
# 4. 决策树分类
# ==============================================
print("\n" + "-" * 70)
print("4. 决策树分类")
print("-" * 70)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_c, y_train_c)
y_pred_dt = dt_clf.predict(X_test_c)

acc_dt = accuracy_score(y_test_c, y_pred_dt)
print(f"\n决策树准确率: {acc_dt:.4f}")
print("\n分类报告:")
print(classification_report(y_test_c, y_pred_dt, target_names=iris.target_names))

# ==============================================
# 5. K最近邻分类
# ==============================================
print("\n" + "-" * 70)
print("5. K最近邻分类")
print("-" * 70)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_c, y_train_c)
y_pred_knn = knn_clf.predict(X_test_c)

acc_knn = accuracy_score(y_test_c, y_pred_knn)
print(f"\nK最近邻准确率: {acc_knn:.4f}")
print("\n分类报告:")
print(classification_report(y_test_c, y_pred_knn, target_names=iris.target_names))

# ==============================================
# 6. 回归任务示例（模拟房价数据）
# ==============================================
print("\n" + "=" * 70)
print("6. 回归任务示例 - 模拟房价数据")
print("=" * 70)

# 生成回归数据
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=8, noise=10.0, random_state=42
)

print(f"\n数据集信息:")
print(f"  特征数据形状: {X_reg.shape}")
print(f"  目标变量形状: {y_reg.shape}")

# 划分训练集和测试集
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ==============================================
# 7. 线性回归
# ==============================================
print("\n" + "-" * 70)
print("7. 线性回归")
print("-" * 70)

lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
y_pred_lin = lin_reg.predict(X_test_r)

mse_lin = mean_squared_error(y_test_r, y_pred_lin)
r2_lin = r2_score(y_test_r, y_pred_lin)

print(f"\n线性回归评估指标:")
print(f"  均方误差 (MSE): {mse_lin:.4f}")
print(f"  R^2 分数: {r2_lin:.4f}")

# 查看系数
print("\n特征系数（权重）:")
feature_names = ['特征0', '特征1', '特征2', '特征3', '特征4', '特征5', '特征6', '特征7']
for name, coef in zip(feature_names, lin_reg.coef_):
    print(f"  {name}: {coef:.4f}")

# ==============================================
# 8. 决策树回归
# ==============================================
print("\n" + "-" * 70)
print("8. 决策树回归")
print("-" * 70)

dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X_train_r, y_train_r)
y_pred_dt_r = dt_reg.predict(X_test_r)

mse_dt = mean_squared_error(y_test_r, y_pred_dt_r)
r2_dt = r2_score(y_test_r, y_pred_dt_r)

print(f"\n决策树回归评估指标:")
print(f"  均方误差 (MSE): {mse_dt:.4f}")
print(f"  R^2 分数: {r2_dt:.4f}")

# ==============================================
# 9. K最近邻回归
# ==============================================
print("\n" + "-" * 70)
print("9. K最近邻回归")
print("-" * 70)

knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_r, y_train_r)
y_pred_knn_r = knn_reg.predict(X_test_r)

mse_knn = mean_squared_error(y_test_r, y_pred_knn_r)
r2_knn = r2_score(y_test_r, y_pred_knn_r)

print(f"\nK最近邻回归评估指标:")
print(f"  均方误差 (MSE): {mse_knn:.4f}")
print(f"  R^2 分数: {r2_knn:.4f}")

# ==============================================
# 10. 算法对比总结
# ==============================================
print("\n" + "=" * 70)
print("10. 算法对比总结")
print("=" * 70)

print("\n【分类任务（鸢尾花数据集）对比】:")
print(f"| 算法         | 准确率 |")
print(f"|--------------|--------|")
print(f"| 逻辑回归     | {acc_log:.4f} |")
print(f"| 决策树       | {acc_dt:.4f} |")
print(f"| K最近邻      | {acc_knn:.4f} |")

print("\n【回归任务（模拟房价数据）对比】:")
print(f"| 算法         | MSE    | R^2分数 |")
print(f"|--------------|--------|---------|")
print(f"| 线性回归     | {mse_lin:.4f} | {r2_lin:.4f} |")
print(f"| 决策树回归   | {mse_dt:.4f} | {r2_dt:.4f} |")
print(f"| K最近邻回归  | {mse_knn:.4f} | {r2_knn:.4f} |")

# ==============================================
# 11. 算法选择建议
# ==============================================
print("\n" + "=" * 70)
print("11. 算法选择建议")
print("=" * 70)
print("""
算法选择指南:
-------------------------------------------------------------
  任务类型        推荐算法          适用场景
  ----------      ----------        ----------
  分类（二分类）   逻辑回归          垃圾邮件检测、疾病诊断
  分类（多分类）   逻辑回归/决策树   图像识别、文本分类
  回归（连续值）   线性回归          房价预测、销量预测
  非线性关系      决策树/KNN        复杂数据模式
  小数据集        KNN               数据量少、特征多
  需要解释性      线性回归/决策树    需要理解模型决策过程
-------------------------------------------------------------

算法特点总结:
  * 线性回归: 简单、快速，适合线性关系
  * 逻辑回归: 输出概率，适合分类问题
  * 决策树: 直观易懂，可处理非线性
  * KNN: 无需训练，对异常值敏感
""")

print("=" * 70)
print("程序执行完成！")
print("=" * 70)