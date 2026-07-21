# ==============================================
# KNN 分类算法示例（详细注释版）
# ==============================================

# ========== 导入必要的库 ==========
# pandas (pd): 数据分析库，用于读取和处理表格数据
import pandas as pd

# train_test_split: 数据划分工具，将数据分为训练集和测试集
from sklearn.model_selection import train_test_split

# StandardScaler: 数据标准化工具，将特征转换为均值=0、标准差=1的分布
from sklearn.preprocessing import StandardScaler

# KNeighborsClassifier: K最近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 评估指标：
# accuracy_score: 计算准确率
# classification_report: 生成分类报告（精确率、召回率、F1分数）
# confusion_matrix: 生成混淆矩阵（详细预测结果）
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========== 1. 加载数据 ==========
# 文件路径（本地CSV文件或网络URL）
url = "SomervilleHappinessSurvey2015.csv"

# pd.read_csv(): 读取CSV文件
#   - 参数1: 文件路径
#   - encoding='utf-8': 文件编码格式（中文文件常用）
df = pd.read_csv(url, encoding='utf-8')

# 打印数据形状（行数, 列数）
print("数据形状:", df.shape)

# 打印前5行数据预览
print(df.head())

# 统计目标列'D'的类别分布（value_counts() 统计每个值出现的次数）
# D列是目标变量，表示幸福指数（0=不满意，1=满意）
print(df['D'].value_counts())

# ========== 2. 数据预处理 ==========
# df.dropna(inplace=True): 删除包含缺失值的行
#   - inplace=True: 直接修改原DataFrame，不创建副本
df.dropna(inplace=True)

# 分离特征和目标变量：
# X: 特征矩阵（所有列除了'D'列）
#   - df.drop('D', axis=1): 删除'D'列
#   - .values: 将DataFrame转换为NumPy数组
X = df.drop('D', axis=1).values

# y: 目标变量（'D'列的值）
y = df['D'].values

# 划分训练集和测试集：
# train_test_split(): 随机划分数据
#   - X, y: 特征和目标变量
#   - test_size=0.3: 测试集占30%，训练集占70%
#   - random_state=42: 随机种子（保证结果可复现）
#   - stratify=y: 分层抽样（保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 数据标准化：
# StandardScaler(): 创建标准化器
scaler = StandardScaler()

# fit_transform(): 先拟合（计算均值和标准差），再转换
#   - 只在训练集上拟合，避免泄露测试集信息
X_train_scaled = scaler.fit_transform(X_train)

# transform(): 只用训练集的参数转换测试集
X_test_scaled = scaler.transform(X_test)

# ========== 3. 训练 KNN 模型 ==========
# KNeighborsClassifier(): 创建KNN分类器
#   - n_neighbors=5: 考虑最近的5个邻居
knn = KNeighborsClassifier(n_neighbors=5)

# fit(): 训练模型（将训练数据传入，模型记住所有数据）
knn.fit(X_train_scaled, y_train)

# predict(): 使用模型进行预测（传入测试集特征）
y_pred = knn.predict(X_test_scaled)

# ========== 4. 评估模型性能 ==========
# accuracy_score(): 计算准确率（预测正确的比例）
acc = accuracy_score(y_test, y_pred)
print(f"\nKNN准确率: {acc:.3f}")

# classification_report(): 生成详细分类报告
#   - 包含每个类别的精确率、召回率、F1分数
print("\n分类报告:\n", classification_report(y_test, y_pred))

# confusion_matrix(): 生成混淆矩阵
#   - 行: 真实类别
#   - 列: 预测类别
#   - 对角线: 正确预测的数量
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))