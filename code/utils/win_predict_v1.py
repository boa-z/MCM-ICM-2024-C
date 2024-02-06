import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_p1 = pd.read_csv("data/output_game_p1_verified.csv")
data_p2 = pd.read_csv("data/output_game_p2_verified.csv")


def Logistic_zzbl_model(data):

    # 读取数据，根据特征权重参数计算胜率
    # 定义特征变量和目标变量
    win_predict = []
    X = data.iloc[:, 3:]

    # 特征数据标准化

    # X_std = (X - X.min()) / (X.max() - X.min())
    X_std = X
    # 0.139  0.165 -0.047 -0.095 -0.025  2.822  0.038
    # win_predict.append((X_std.iloc[:, 0] * 0.139 + X_std.iloc[:, 1] * 0.165 + X_std.iloc[:, 2] * -0.047 +
    #                    X_std.iloc[:, 3] * -0.095 + X_std.iloc[:, 4] * -0.025 + X_std.iloc[:, 5] * 2.822 + X_std.iloc[:, 6] * 0.038)/3)

    # 2.502 -0.019 -0.015  0.019 -0.015  0.016  2.502 -0.01
    win_predict.append((X_std.iloc[:, 0] * 2.502 + X_std.iloc[:, 1] * -0.019 + X_std.iloc[:, 2] * -0.015 + X_std.iloc[:, 3]
                       * 0.019 + X_std.iloc[:, 4] * -0.015 + X_std.iloc[:, 5] * 0.016 + X_std.iloc[:, 6] * 2.502 + X_std.iloc[:, 7] * -0.01)/4)
    # 0.131 0.411 0.146 0.366 1.062 1188.000 0.569 (SPSS 之权重，鉴定为不如)
    # win_predict.append((X_std.iloc[:, 0] * 0.131 + X_std.iloc[:, 1] * 0.411 + X_std.iloc[:, 2] * 0.146 + X_std.iloc[:, 3] * 0.366 +
    #                       X_std.iloc[:, 4] * 1.062 + X_std.iloc[:, 5] * 1188.000 + X_std.iloc[:, 6] * 0.569)/3)
    
    return win_predict


win_predict_p1 = Logistic_zzbl_model(data_p1)
win_predict_p2 = Logistic_zzbl_model(data_p2)

# # 绘制胜率图（折线图）
plt.plot(win_predict_p1[0], label='player1')
plt.plot(win_predict_p2[0], label='player2')
plt.xlabel('比赛场数')
plt.ylabel('胜率')
plt.show()
