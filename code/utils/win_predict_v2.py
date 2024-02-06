import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_path = "data/Wimbledon_featured_matches/Standard_Training_Data_Wimbledon_featured_matches_match_id.csv"
data_p1 = pd.read_csv(data_path)


def win_predict_LR_model(data) -> pd.DataFrame:

    # 读取数据，根据特征权重参数计算胜率
    # 定义特征变量和目标变量
    win_predict = []
    X = data

    # 特征数据标准化

    # X_std = (X - X.min()) / (X.max() - X.min())
    X_std = X

    # -1.761 0.081 -0.082 0.408 -0.223 -0.036 0.379 0.613 0.183 0.093
    win_predict.append((X_std.iloc[:, 0] * -1.761 + X_std.iloc[:, 1] * 0.081 + X_std.iloc[:, 2] * -0.082 + X_std.iloc[:, 3] * 0.408 + X_std.iloc[:, 4] * -0.223 + X_std.iloc[:, 5] * -0.036 + X_std.iloc[:, 6] * 0.379 + X_std.iloc[:, 7] * 0.613 + X_std.iloc[:, 8] * 0.183 + X_std.iloc[:, 9] * 0.093))

    # 将胜率数据写入文件的最后一列，将包含胜率数据的DataFrame返回
    data['win_predict'] = win_predict[0]
    
    return data

# win_predict_p1 = win_predict_LR_model(data_p1)
# print(data_p1)
# # 将胜率数据写入文件的最后一列，保存为csv文件
# data_p1['win_predict'] = win_predict_p1[0]
# data_p1.to_csv(data_path.replace('.csv', '_win_predict.csv'), index=False)

# # 按照 match_id 进行分组并画图
# list_match = data_p1['match_id'].unique()

# for match in list_match:
#     # 读取数据
#     match_data = data_p1[data_p1['match_id'] == match]
#     # x11 = match_data['x11']
#     # win_predict = match_data['win_predict']

#     # 绘制图表
#     plt.figure(figsize=(10, 6))
#     plt.plot(match_data['x11'], label='x11')
#     plt.plot(match_data['win_predict'], label='win_predict')

#     plt.title(f'Win Prediction for Match {match}')
#     plt.xlabel('Data Point')
#     plt.ylabel('Win Prediction')
#     plt.legend()
#     plt.savefig(f'figure/win_prediction_{match}.png', dpi=500)
#     plt.close()
#     print(f'figure/win_prediction_{match}.png has been saved.')
