# 对model_setup_LR_v2.py建立的模型进行敏感性分析
# 使用 SALib 库进行全局敏感性分析

import pandas as pd
import numpy as np
import json
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt

# 读取config.json文件
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

# 读取数据集
data_path = config_data["lr_v2_data_path"]

# 为数据集添加列名
zzbl_data = pd.read_csv(data_path)
zzbl_data.head()

# 创建一个LabelEncoder对象

le = LabelEncoder()

# 对每一列进行标签编码
for col in zzbl_data.columns:
    zzbl_data[col] = le.fit_transform(zzbl_data[col])
zzbl_data.head()

# 定义特征变量和目标变量
# X 参考特征变量：
# sever, serve_no, p1_points_won, p1_winner, p1_double_fault, p1_unf_err
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# 特征数据标准化
scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)

X_test_std = scaler.transform(X_test)

print(X_train_std, X_test_std)

# 创建并训练逻辑回归模型

model = LogisticRegression(C=1.0, solver="lbfgs", multi_class="auto", max_iter=1000, random_state=0)

model.fit(X_train_std, y_train)

# 计算胜率

print("学得的特征权重参数：\n", np.round(model.coef_, 3), sep="")
print("学得的模型截距：", np.round(model.intercept_, 3))
print("样本类别：", model.classes_)
# 对model_setup_LR_v2.py建立的模型进行敏感性分析
# 使用 SALib 库进行全局敏感性分析

# 定义问题
problem = {
    'num_vars': 6,
    'names': ['sever', 'serve_no', 'p1_points_won', 'p1_winner', 'p1_double_fault', 'p1_unf_err'],
    'bounds': [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
}

# 生成样本
param_values = saltelli.sample(problem, 1000)

# 运行模型
Y = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    Y[i] = model.predict([X])

# 分析结果
Si = sobol.analyze(problem, Y)

# 输出结果
print(Si)

# 可视化
fig, ax = plt.subplots(1)

Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
Si_df = pd.DataFrame(Si_filter, index=problem['names'])

Si_df.plot(kind='bar', y='ST', yerr='ST_conf', ax=ax, color='lightblue')
Si_df.plot(kind='bar', y='S1', yerr='S1_conf', ax=ax, color='orange')

plt.show()


# 保存模型
joblib.dump(model, "lr_model.pkl")

# 保存模型性能指标

model_performance = {
    "Accuracy": model.score(X_test_std, y_test),
    "Sensitivity Analysis": Si['ST'].tolist(),  # Convert 'ResultDict' object to list
}

with open("lr_model_performance.json", "w") as json_file:
    json.dump(model_performance, json_file)

