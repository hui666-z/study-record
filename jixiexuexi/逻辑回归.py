'''
场景：银行要不要给这个客户批贷款
假设你是银行风控经理，手里有 100 个客户的历史记录，
每个客户有两个特征：【年收入（万）】和【负债率（%）】，标签是【是否违约】（0=不违约，1=违约）。
'''

import numpy as np
from  sklearn.linear_model import  LogisticRegression  # 逻辑回归
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

# step 准备数据
X = np.random.rand(100,2) * 100 # 随机生成 100条数据，每条数据也有两特征，收入范围是在 0-100  负债率是在 0-100
# 规则：收入 > 50 且负载率 <50 ==> 不违规（0） 否则违约（1）
y = 1 - ((X[:,0]>50) & (X[:,1]<50)).astype(int)
# print(X[:10])
# print(y[:10])

# 切分数据集  因为生成的数据类别已经是乱序的，所以在切分的时候，可以会需要随机切分
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# 训练模型
model = LogisticRegression()
model.fit(x_train,y_train)


# 评估模型
y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# 预测新客户
new_customer = np.array([[20,50]])
prob = model.predict_proba(new_customer)
print(f"不违约概率：{prob[0][0]:.2f},违约的概率：{prob[0][1]:.2f}")

