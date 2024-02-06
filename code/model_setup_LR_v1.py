import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# 逻辑回归模型
# 编写一个用于训练逻辑回归模型的函数
def train_rf_model(data, test_size=0.3, random_state=42, n_estimators=100, model_file_name='rf_model.pkl'):
    # 划分特征和标签
    X = data.iloc[:, 0:10]
    print(X)
    y = data['label']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 创建逻辑回归分类器
    lr_classifier = LogisticRegression(C=1.0, solver="lbfgs", multi_class="auto", max_iter=1000, random_state=0)

    # 在训练集上训练模型
    lr_classifier.fit(X_train, y_train)

    # 输出学得的特征权重参数等信息
    print("学得的特征权重参数：\n", np.round(lr_classifier.coef_, 3), sep="")
    learned_feature_weight_parameters = np.round(lr_classifier.coef_, 3)
    print("学得的模型截距：", np.round(lr_classifier.intercept_, 3))
    print("样本类别：", lr_classifier.classes_)
    print("训练集准确率：", round(lr_classifier.score(X_train, y_train), 3))
    print("测试集准确率：", round(lr_classifier.score(X_test, y_test), 3))

    # 在测试集上进行预测
    y_pred = lr_classifier.predict(X_test)

    # 计算模型性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Logistic Regression Model Performance:")
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"R² Score: {r2:.2f}")

    # 计算Pearson相关系数
    corr_coef, p_value = stats.pearsonr(y_pred, y_test)
    print("Pearson相关系数为:", round(corr_coef,3))
    print("p值为:", p_value)

    # 计算各个指标的显著性
    y_xzx = y_test
    x_xzx = y_pred
    X_xzx = X
    X_xzx = sm.add_constant(x_xzx)
    model = OLS(y_xzx,X_xzx,fit_intercept=True).fit()
    print(model.summary())

    # 保存模型
    joblib.dump(lr_classifier, model_file_name)
    print(f"Model saved to {model_file_name}")

    # 将模型性能指标保存到 json 文件中
    model_performance = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "R² Score": r2
    }

    return lr_classifier, X, y, X_train, X_test, y_train, y_test, model_performance, learned_feature_weight_parameters

# # 对完整比赛中选手表现进行预测，将预测结果保存到列表中，要求调用 train_rf_model 函数，保留小数点后两位
# def match_performance_pred(data, new_file_name_pred, feature_importance_file_name, model_performance_file_name):

#     # 导入模型
#     lr_classifier = train_rf_model(data)[0]
#     X = train_rf_model(data)[1]

#     # 预测比赛中选手表现
#     y_pred = lr_classifier.predict(X)

#     # 将预测结果保存到文件中
#     y_pred_df = pd.DataFrame(y_pred, columns=['performance'])
#     y_pred_df.to_csv(new_file_name_pred, index=False)

#     # 输出特征重要性
#     feature_importance = pd.DataFrame({
#         "Feature": X.columns,
#         "Importance": lr_classifier.coef_[0]
#     })
#     feature_importance.to_csv(feature_importance_file_name, index=False)

#     # 输出模型性能指标
#     model_performance = train_rf_model(data)[7]
#     with open(model_performance_file_name, 'w') as file:
#         json.dump(model_performance, file)

#     return y_pred_df, feature_importance, model_performance

# 对完整比赛中选手表现进行计算，将计算结果保存到列表中新的一列，命名为 match_performance，要求用 train_rf_model 函数计算得到的 learned_feature_weight_parameters 去计算data中的X的加权和，保留小数点后两位
def match_performance_calc(data, new_file_name_pred, feature_importance_file_name, model_performance_file_name):

    # 导入模型
    lr_classifier = train_rf_model(data)[0]
    X = train_rf_model(data)[1]
    learned_feature_weight_parameters = train_rf_model(data)[8]

    # 计算比赛中选手表现
    y_pred = lr_classifier.predict(X)

    # 将预测结果保存到文件中
    y_pred_df = pd.DataFrame(y_pred, columns=['performance'])
    y_pred_df.to_csv(new_file_name_pred, index=False)

    # 输出特征重要性
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": lr_classifier.coef_[0]
    })
    feature_importance.to_csv(feature_importance_file_name, index=False)

    # 输出模型性能指标
    model_performance = train_rf_model(data)[7]
    with open(model_performance_file_name, 'w') as file:
        json.dump(model_performance, file)

    # 计算比赛中选手表现
    match_performance = np.dot(X, learned_feature_weight_parameters.T)
    match_performance_df = pd.DataFrame(match_performance, columns=['match_performance'])
    data_with_match_performance = pd.concat([data, match_performance_df], axis=1)
    data_with_match_performance.to_csv(new_file_name_pred, index=False)
    
    # # 将'match_performance'和'label'以折线图的形式展示
    # plt.plot(data_with_match_performance['match_performance'], label='match_performance')
    # plt.plot(data_with_match_performance['label'], label='label')
    # plt.legend()
    # plt.show()

    return y_pred_df, feature_importance, data_with_match_performance

###################################

# 导入并画出 train_rf_model 函数训练的模型，并画出混淆矩阵
def plot_confusion_matrix(data, model_file_name):
    # 导入模型
    lr_classifier = joblib.load(model_file_name)

    # 划分特征和标签
    X = data.iloc[:, 0:10]
    y = data['label']

    # 在测试集上进行预测
    y_pred = lr_classifier.predict(X)

    # 画出混淆矩阵
    cm = confusion_matrix(y, y_pred)
    print(cm)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')  # Include actual values in the heatmap plot
    plt.savefig('confusion_matrix.png')

# 读取config.json文件
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

# 读取数据集
data_path = config_data["lr_v1_data_path"]

# 为数据集添加列名

zzbl_data = pd.read_csv(data_path)
zzbl_data.head()

# # 测试 train_rf_model 函数
# train_rf_model(zzbl_data)

# 测试 match_performance_pred 函数、
# match_performance_pred(zzbl_data, 'lr_v1_data_pred.csv', 'lr_v1_feature_importance.csv', 'lr_v1_model_performance.json')

# match_performance_calc(zzbl_data, 'lr_v1_data_pred.csv', 'lr_v1_feature_importance.csv', 'lr_v1_model_performance.json')

# 测试 plot_confusion_matrix 函数
# plot_confusion_matrix(zzbl_data, 'rf_model.pkl')
