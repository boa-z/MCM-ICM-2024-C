import pandas as pd
import numpy as np
import scipy.stats as stats
import json

# 读取config.json文件
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

# 读取数据集
data_path = config_data["lr_v2_data_path"]

# 为数据集添加列名

zzbl_data = pd.read_csv(data_path)
zzbl_data.head()

from sklearn.preprocessing import LabelEncoder
# 创建一个LabelEncoder对象
le = LabelEncoder()
# 对每一列进行标签编码
for col in zzbl_data.columns:
    zzbl_data[col] = le.fit_transform(zzbl_data[col])
zzbl_data.head()

# 定义特征变量和目标变量
# X 参考特征变量：
# sever, serve_no, p1_points_won, p1_winner, p1_double_fault, p1_unf_err
# 即为csv文件中的第14, 15, 23, 26, 28 列
X = zzbl_data.iloc[:, [13, 14, 16, 22, 25, 27]]
print(X)
# y = zzbl_data['label']
# y 参考目标变量：point_victor = 1, 2, 需要转换为0, 1
y = zzbl_data['point_victor']
print(y)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# 特征数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) 
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

print(X_train_std, X_test_std)

# 创建并训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, solver="lbfgs", multi_class="auto", max_iter=1000, random_state=0)
model.fit(X_train_std, y_train) 

# 计算胜率

print("学得的特征权重参数：\n", np.round(model.coef_, 3), sep="")
print("学得的模型截距：", np.round(model.intercept_, 3))
print("样本类别：", model.classes_)

# 性能评估
print("训练集准确率：", round(model.score(X_train_std, y_train), 3))
print("测试集准确率：", round(model.score(X_test_std, y_test), 3))

# 预测测试集数据
y_test_pred = model.predict(X_test_std)
print("预测的测试集数据标签前3项：", y_test_pred[:3])

# 我们使用皮尔逊相关系数检验分析相关系
# 计算Pearson相关系数

corr_coef, p_value = stats.pearsonr(y_test_pred, y_test)
print("Pearson相关系数为:", round(corr_coef,3))
print("p值为:", p_value)

from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

y = y_test
x = y_test_pred
X = sm.add_constant(x)
model = OLS(y,X,fit_intercept=True).fit()
print(model.summary())

# # kaka: 不可以的
# # 混淆矩阵
# from sklearn.metrics import confusion_matrix
# c_matrix = confusion_matrix(y_test, y_test_pred)
# import seaborn as sns
# import matplotlib.pyplot as plt
# ax = plt.subplot()
# sns.heatmap(c_matrix, annot=True, fmt='g', cmap='Blues', ax=ax)
# ax.set_title('Confusion Matrix')
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.xaxis.set_ticklabels(le.classes_)
# ax.yaxis.set_ticklabels(le.classes_)
# # 显示图形
# plt.show()
