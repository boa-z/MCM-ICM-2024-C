import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json

# 编写一个用于训练随机森林模型的函数

def train_rf_model(data, test_size=0.3, random_state=42, n_estimators=100, model_file_name='rf_model.pkl'):
    # 划分特征和标签
    X = data.iloc[:, :-3]  # x1-x11的值
    y = data['turning_point']  # 是否为1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # 在训练集上训练模型
    rf_classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rf_classifier.predict(X_test)

    # 计算模型性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Model Performance:")
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"R² Score: {r2:.2f}")

    # # 保存模型
    # joblib.dump(rf_classifier, model_file_name)
    # print(f"Model saved to {model_file_name}")
    
    # 将模型性能指标保存到 json 文件中
    model_performance = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "R² Score": r2
    }

    return rf_classifier, X, y, X_train, X_test, y_train, y_test, model_performance

# 对完整比赛进行拐点检测，将预测结果保存到列表中，调用 train_rf_model 函数

def match_turning_point_pred(data, new_file_name_pred, feature_importance_file_name, model_performance_file_name):

    # 导入模型
    rf_classifier = train_rf_model(data)[0]
    X = train_rf_model(data)[1]

    turning_points_pred = []

    # 按照比赛进行拐点检测
    # x1-x11的值
    X_match = data.iloc[:, :-3]
    y_match = data['turning_point']  # 是否为1

    # 预测
    y_pred = rf_classifier.predict(X_match)
    turning_points_pred.append(y_pred)

    # 将预测结果保存到列表中，并写入文件
    data_with_turning_points = data.copy()
    data_with_turning_points['turning_point_pred'] = y_pred
    # 保存到 csv 文件中，新文件命名为原文件名加上 '_pred'
    # data_with_turning_points.to_csv(data_path.replace('.csv', '_pred.csv'), index=False)

    data_with_turning_points.to_csv(new_file_name_pred, index=False)
    # print(f"Prediction for turning points saved to {data_path.replace('.csv', '_pred.csv')}")
    print(f"Prediction for turning points saved to {new_file_name_pred}")

    # 输出特征重要性得分
    match_feature_importances = rf_classifier.feature_importances_
    match_feature_importances_df = pd.DataFrame({'Feature Name': X.columns, 'Feature Importance': match_feature_importances})
    match_feature_importances_df.to_csv(feature_importance_file_name, index=False)

    # 输出模型性能指标
    model_performance = train_rf_model(data)[7]
    with open(model_performance_file_name, 'w') as json_file:
        json.dump(model_performance, json_file)

# 进行随机森林特征重要性检测，计算特征重要性得分

def feature_importance_calculation(data) -> pd.DataFrame:

    # 导入模型
    rf_classifier = train_rf_model(data)[0]

    X = data.iloc[:, :-3]  # x1-x11的值
    y = data['turning_point']  # 是否为1

    # 获取特征重要性得分
    feature_importances = rf_classifier.feature_importances_

    feature_importances_df = pd.DataFrame({'Feature Name': X.columns, 'Feature Importance': feature_importances})

    return feature_importances_df

# # 测试
# # 读取config.json文件
# with open('config.json', 'r') as config_file:
#     config_data = json.load(config_file)

# # 读取数据集
# data_path = config_data["rf_data_path"]
# data = pd.read_csv(data_path)  # 替换为你的数据文件路径

# train_rf_model(data)
# # match_turning_point_pred(data, 'test1.csv', 'test2.csv')
# # feature_importance_divided_by_match_id(data)
