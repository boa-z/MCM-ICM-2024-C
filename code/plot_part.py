# 模型检测

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re


class plot_part:
    def __init__(self, data):
        self.data = data

    def rf_turning_point_pred(self):
        # 需要的数据文件路径形如：data/{origin_file_name}/Standard_Training_Data_{origin_file_name}_match_id_pred.csv

        list_match = self.data['match_id'].unique()

        for match in list_match:
            # 读取数据
            match_data = data[data['match_id'] == match]
            x11 = match_data['x11']
            turning_points = match_data[match_data['turning_point'] == 1].index
            turning_points_pred = match_data[match_data['turning_point_pred'] == 1].index

            # 绘制图表
            plt.figure(figsize=(10, 6))
            # 使用 sns 的 lineplot 函数绘制折线图，并试图平滑曲线
            sns.set(style="whitegrid")
            sns.lineplot(x=match_data.index, y=x11, label='momentum')
            # sns.scatterplot(x=turning_points, y=[x11[i] for i in turning_points], label='True Turning Point', marker='o', color='green', palette="deep") 
            # sns.scatterplot(x=turning_points_pred, y=[x11[i] for i in turning_points_pred], label='Predicted Turning Point', marker='*', color='red', palette="deep")
            # # plt.plot(match_data['x11'], label='momentum')

            plt.scatter(turning_points, [x11[i] for i in turning_points], color='xkcd:orange', label='True Turning Point', marker='o')
            # plt.scatter(turning_points_pred, [x11[i] for i in turning_points_pred], color='xkcd:light green', label='Predicted Turning Point', marker='x')

            # 获取当前的坐标轴对象
            ax = plt.gca()

            # 调整图层的顺序，让第一组数据在第二组数据的上方
            ax.collections[0].set_zorder(1)
            ax.collections[1].set_zorder(0)

            plt.title(f'Turning Point Detection for Match {match}')
            plt.xlabel('Data Point')
            plt.ylabel('Turning Point')
            plt.legend()
            plt.savefig(f'figure/{origin_file_name}/turning_point_detection_{match}_prediction.png', dpi=500)
            plt.close()
            print(f'figure/{origin_file_name}/turning_point_detection_{match}_prediction.png has been saved.')

    # 画出整场比赛的feature_importance图
    # 数据来源是 data
    def rf_match_feature_importance(self, nmsl='feature_importance_Wimbledon_featured_matches.csv'):

        # 需要的数据文件路径形如：data/{origin_file_name}/feature_importance_{match_id}.csv
        data = pd.read_csv(nmsl)
        list_match = data['match_id'].unique()

        for match in list_match:
            # 读取数据
            match_data = data[data['match_id'] == match]
            feature_importance = match_data['feature_importance']
            feature_name = match_data['feature_name']

            # 绘制图表
            plt.figure(figsize=(10, 6))
            # 使用 sns 的 barplot 函数绘制柱状图
            sns.set(style="whitegrid")
            sns.barplot(x=feature_name, y=feature_importance, palette="deep")

            plt.title(f'Feature Importance for Match {match}')
            plt.xlabel('Feature Name')
            plt.ylabel('Feature Importance')
            plt.savefig(f'figure/{origin_file_name}/feature_importance_{match}.png', dpi=500)
            plt.close()
            print(f'figure/{origin_file_name}/feature_importance_{match}.png has been saved.')

    # 画出每一位球员的feature_importance的饼图

    def rf_player_feature_importance(self, player_in_match_json_path):

        # 需要的数据文件路径形如：data/{origin_file_name}/feature_importance_{match_id}.csv
        # 读取player_in_match.json文件
        # 需要的 json 文件路径形如：data/{origin_file_name}/player_in_match_{origin_file_name}.json

        with open(player_in_match_json_path, 'r') as json_file:
            players_matches = json.load(json_file)

        # 创建一个空字典来存储每一位球员的feature_importance数据
        players_feature_importance = {}

        # 遍历每一位球员
        for player, match_ids in players_matches.items():
            # 初始化一个空的DataFrame来存储feature_importance数据
            player_data = pd.DataFrame(columns=['Feature Name', 'Feature Importance'])

            # 遍历球员参与的每场比赛
            for match_id in match_ids:
                # 构建feature_importance文件路径
                file_path = f'data/{origin_file_name}/feature_importance_{match_id}.csv'

                # 读取CSV文件
                if os.path.exists(file_path):
                    match_data = pd.read_csv(file_path)
                    player_data = pd.concat([player_data, match_data], ignore_index=True)

            # 按照Feature Name进行分组并计算平均值
            player_avg_feature_importance = player_data.groupby('Feature Name')['Feature Importance'].mean().reset_index()

            # 存储每一位球员的平均feature_importance数据
            players_feature_importance[player] = player_avg_feature_importance

        # 绘制饼图，使用 sns 柔和配色，对饼图中占比最大的部分突出显示
        for player, feature_importance_data in players_feature_importance.items():
            plt.figure(figsize=(8, 8))

            # 设置柔和的配色
            colors = sns.color_palette('pastel')
            
            sns.set(style="whitegrid")  # 设置图表样式
            plt.pie(feature_importance_data['Feature Importance'], labels=feature_importance_data['Feature Name'], autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
            # sns.pairplot(feature_importance_data, x_vars='Feature Importance', y_vars='Feature Name', kind='bar', palette='pastel')
            
            plt.title(f'Feature Importance for {player}')
            plt.savefig(f'figure/{origin_file_name}/feature_importance_{player}.png', dpi=500)
            plt.close()
            print(f'figure/{origin_file_name}/feature_importance_{player}.png has been saved.')

    # 画出标准化后的momentum和scoregap的关系，将两条线画在一张折线图
    # momentum 和 scoregap 分别对应 x11 和 x1，由于 x11 和 x1 绝对值差距较大，所以需要标准化
    # 数据来源是 data 
    # 按照 match_id 进行分组，并画图
    def momentum_relation_with_scoregap(self):

        # 需要的数据文件路径形如：data/{origin_file_name}/Standard_Training_Data_{origin_file_name}_match_id.csv

        list_match = data['match_id'].unique()

        for match in list_match:
            # 读取数据
            match_data = data[data['match_id'] == match]
            x1 = match_data['x1']
            x11 = match_data['x11']

            # 标准化
            x1 = (x1 - x1.mean()) / x1.std()
            x11 = (x11 - x11.mean()) / x11.std()

            # 绘制图表
            sns.set(style="whitegrid")  # 设置图表样式

            # 绘制为散点图
            plt.figure(figsize=(10, 6))
            # sns.scatterplot(x=match_data.index, y=x1, label='Standardized Score Gap', color='blue')
            # sns.scatterplot(x=match_data.index, y=x11, label='Standardized Momentum', color='red')

            sns.lineplot(x=match_data.index, y=x1, label='Standardized Score Gap')
            sns.lineplot(x=match_data.index, y=x11, label='Standardized Momentum')


            plt.title(f'Standardized Momentum and Score Gap for Match {match}')
            plt.xlabel('Data Point')
            plt.ylabel('Standardized Value')
            plt.legend()
            plt.savefig(f'figure/{origin_file_name}/standardized_momentum_scoregap_{match}.png', dpi=500)
            plt.close()

            # plt.figure(figsize=(10, 6))
            # plt.plot(x1, label='Standardized Score Gap')
            # plt.plot(x11, label='Standardized Momentum')

            # plt.title(f'Standardized Momentum and Score Gap for Match {match}')
            # plt.xlabel('Data Point')
            # plt.ylabel('Standardized Value')
            # plt.legend()
            # plt.savefig(f'figure/{origin_file_name}/standardized_momentum_scoregap_{match}.png', dpi=500)
            # plt.close()
            # print(f'figure/{origin_file_name}/standardized_momentum_scoregap_{match}.png has been saved.')

    
    def match_performance_pred(self):
        
        list_match = data['match_id'].unique()

        for match in list_match:

            match_data = data[data['match_id'] == match]
            match_performance = match_data['match_performance']

            # 绘制图表
            sns.set(style="whitegrid")  # 设置图表样式
            plt.figure(figsize=(10, 6)) # 设置图表大小
            # 绘制为折线图
            # sns.lineplot(x=match_data.index, y=match_performance, label='match_performance', color='green')
            # 绘制为柱状图
            # sns.barplot(x=match_data.index, y=match_performance, label='match_performance', color='green')
            # 绘制为直方图
            # sns.histplot(match_performance, kde=True, label='match_performance', color='green')
            
            colors = sns.color_palette('pastel')
            # 每10个数取平均值
            rolling_mean_10 = match_data['match_performance'].rolling(window=10).mean()
            sns.lineplot(x=match_data.index, y=rolling_mean_10, label='match_performance_ma10')
            # 绘制平均值为5的移动平均线，并且为平滑曲线
            rolling_mean_5 = match_data['match_performance'].rolling(window=5).mean()
            sns.lineplot(x=match_data.index, y=rolling_mean_5, label='match_performance_ma5')
            
            # 绘制为散点图
            # sns.scatterplot(x=match_data.index, y=match_performance, label='match_performance', color='green')

            # sns.lineplot(x=match_data.index, y=rolling_mean, label='match_performance_ma10')

            plt.title(f'Match Performance for Match {match}')
            plt.xlabel('Data Point')
            plt.ylabel('Match Performance')
            plt.legend()
            plt.savefig(f'figure/{origin_file_name}/match_performance_{match}.png', dpi=500)
            plt.close()
            print(f'figure/{origin_file_name}/match_performance_{match}.png has been saved.')


###########################################################

# 读取config.json文件
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

# 读取数据集
origin_data_path = config_data["origin_data_path"]

origin_file_name = re.findall(r'[^/]+(?=\.xlsx|\.csv)', origin_data_path)[0]

# 判断文件夹是否存在，不存在则创建
if not os.path.exists('figure/' + origin_file_name):
    os.makedirs('figure/' + origin_file_name)

# data_path = config_data["lr_v1_data_path"]
# data = pd.read_csv(data_path)

# plot_part = plot_part(data)
# plot_part.momentum_relation_with_scoregap()

data_path = config_data["rf_result_data_path"]
data = pd.read_csv(data_path)

plot_part = plot_part(data)
# plot_part.rf_turning_point_pred()
# plot_part.rf_match_feature_importance()
player_in_match_json_path = config_data["player_in_match"]
plot_part.rf_player_feature_importance(player_in_match_json_path)


# data_path = config_data["lr_v1_result_data_path"]

# data = pd.read_csv(data_path)
# plot_part_instance = plot_part(data)
# plot_part_instance.match_performance_pred()

# ###########################################################

# # 编写函数，输入为要用于绘图的数据集的路径（str，不需要从json中读取），根据data_cleanup_and_generate.py的函数，尝试自动寻找处理后用于绘图的数据集
# # 如何找到，则返回处理后数据的路径，否则返回None

# def find_data_for_plot(data_path: str) -> list:

#     data_after_processing_path = []

#     # 检查class plot_part中各个函数需要的数据文件是否存在
#     origin_file_name = re.findall(r'[^/]+(?=\.xlsx|\.csv)', data_path)[0]
#     data_dir = 'data' + '/' + origin_file_name + '/'
#     print(data_dir)

#     # 1. Standard_Training_Data_{origin_file_name}_match_id_pred.csv
#     for file in os.listdir(data_dir):
#         if re.match(rf'Standard_Training_Data_{origin_file_name}_\d+_pred.csv', file):
#             data_after_processing_path.append(data_dir + file)
#             print(data_dir + file + ' has been found.')
#         else:
#             data_after_processing_path.append(None)
#             print(data_dir + file + ' has not been found.')

    
#     # 2. feature_importance_{match_id}.csv
#     for file in os.listdir(data_dir):
#         if re.match(rf'feature_importance_\d+.csv', file):
#             data_after_processing_path.append(data_dir + file)
#         else:
#             data_after_processing_path.append(None)
#             print(data_dir + file + ' has not been found.')

#     # 3. Standard_Training_Data_{origin_file_name}_match_id.csv
#     for file in os.listdir(data_dir):
#         if re.match(rf'Standard_Training_Data_{origin_file_name}_\d+.csv', file):
#             data_after_processing_path.append(data_dir + file)
#         else:
#             data_after_processing_path.append(None)
#             print(data_dir + file + ' has not been found.')

#     # 4. Standard_Training_Data_{origin_file_name}_match_id_win_pred.csv
#     for file in os.listdir(data_dir):
#         if re.match(rf'Standard_Training_Data_{origin_file_name}_\d+_win_pred.csv', file):
#             data_after_processing_path.append(data_dir + file)
#         else:
#             data_after_processing_path.append(None)
#             print(data_dir + file + ' has not been found.')

#     # 将以上路径存入列表中

#     return data_after_processing_path

# # 测试运行
    
# test = find_data_for_plot(origin_data_path)
# print(test)
