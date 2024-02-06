'''
基于九条之代码，补充一些要计算的数值
生成数据后，请使用 SPSS 进行数据分析
'''


from sklearn.preprocessing import MinMaxScaler
import openpyxl
import pandas as pd
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import re
import json
import numpy as np
import os

from utils.data_set_turning_point_v2 import calculate_turning_points
from model_setup_random_forest import match_turning_point_pred, feature_importance_calculation
from utils.win_predict_v2 import win_predict_LR_model
from model_setup_LR_v1 import match_performance_calc


class DataCleanAndGenerate:
    def __init__(self, df_path):
        self.df_path = df_path
        self.df = self.data_clean(df_path)
        # self.data_with_turning_points = self.calculate_turning_points()[1]
        # self.turning_points_pred = self.match_turning_point_pred()

    # Samples with abnormal scores were eliminated, and samples with missing pace were deleted.
    def data_clean(self, df_path):

        # Select different reading methods based on the df_path suffix name
        if df_path.endswith('.xlsx'):
            df = pd.read_excel(df_path)
        elif df_path.endswith('.csv'):
            df = pd.read_csv(df_path)

        df.loc[(df.p1_score == 'AD'), 'p1_score'] = 50
        df.loc[(df.p2_score == 'AD'), 'p2_score'] = 50
        df['p1_score'] = df['p1_score'].astype(int)
        df['p2_score'] = df['p2_score'].astype(int)
        df.dropna(subset=['speed_mph'], inplace=True)

        # Save the cleaned data to a new folder

        data_dir = self.get_new_folder_name()
        new_file_name = self.get_origin_file_name().split('.')[0] + '_edit.csv'
        new_file_name = os.path.join(data_dir, new_file_name)

        df.to_csv(new_file_name, index=False)
        print('file path after clean:', new_file_name)

        return df

    # 通过正则表达式识别 df_path 中的文件名，不包含文件后缀，也不包含路径，随后在data/文件夹中创建一个新的文件夹，文件夹名为文件名
    def get_origin_file_name(self) -> str:

        folder_name = re.findall(r'[^/]+(?=\.xlsx|\.csv)', self.df_path)[0]

        # 判断文件夹是否存在，不存在则创建
        if not os.path.exists('data/' + folder_name):
            os.makedirs('data/' + folder_name)

        return re.findall(r'[^/]+(?=\.xlsx|\.csv)', self.df_path)[0]

    def get_new_folder_name(self) -> str:

        folder_name = 'data/' + \
            re.findall(r'[^/]+(?=\.xlsx|\.csv)', self.df_path)[0]

        # 判断文件夹是否存在，不存在则创建
        if not os.path.exists('data/' + folder_name):
            os.makedirs('data/' + folder_name)

        return folder_name

    def calculate_comprehensive_momentum(self, player_number, window_size=4) -> list:

        data = self.df
        momentum_scores = [0] * len(data)
        consecutive_point_wins = 0  # 追踪连续得分
        consecutive_game_wins = 0  # 追踪连续获胜的局数
        previous_game_winner = None  # 追踪上一局的获胜者
        initial_break_point_value = 1  # 破发的基础势头得分增加值

        for i in range(1, len(data)):
            recent_data = data[max(0, i - window_size):i]
            momentum_score = 0

            for _, feature in recent_data.iterrows():
                # 基本的势头得分计算
                P_t = 1 if feature['point_victor'] == player_number else -1
                S_t = 1.2 if feature['server'] == player_number else 1.0
                base_momentum = P_t * S_t
                momentum_score += base_momentum
                break_point_value = initial_break_point_value  # 重置破发得分值

                # 连续得分补正（线性）
                if P_t == 1:
                    consecutive_point_wins += 1
                else:
                    consecutive_point_wins = 0  # 在失分时重置
                momentum_score += 0.03 * consecutive_point_wins  # 每连续获胜增加额外得分

                # 连续小局获胜补正（线性）
                if feature['game_victor']:
                    current_game_winner = feature['game_victor']
                    if current_game_winner == player_number:
                        if current_game_winner == previous_game_winner:
                            consecutive_game_wins += 1
                        else:
                            consecutive_game_wins = 0  # 重置连续获胜局数
                    previous_game_winner = current_game_winner
                    momentum_score += 0.2 * consecutive_game_wins  # 连续获胜局数的影响

                # 大比分差距补正（指数）
                if feature['set_victor']:
                    player1_set = feature['p1_sets'] + \
                        1 if feature['set_victor'] == player_number else feature['p1_sets']
                    player2_set = feature['p2_sets'] + \
                        1 if feature['set_victor'] == player_number else feature['p2_sets']
                    diff = (player2_set - player1_set) * \
                        (-1 ** player_number)  # player1为-1， player2为+1
                    momentum_score += 0.1 * (2 ** diff)

                # 小比分差距补正（线性）
                if feature['game_victor']:
                    score_diff = abs(feature['p1_games'] - feature['p2_games'])
                    momentum_score += 0.02 * score_diff * P_t

                # 错失破发点对破发的势头得分增加值的削弱
                if feature['p1_break_pt_missed'] == 1 or feature['p2_break_pt_missed'] == 1:
                    break_point_value -= 0.1  # 削弱的权值

                # (被)破发的影响
                if feature['p1_break_pt_won'] == 1 or feature['p2_break_pt_won'] == 1:
                    break_point_value = max(break_point_value, 0.1)
                    momentum_score += break_point_value * P_t

                # 拍数和跑动距离的影响
                rally_factor = feature['rally_count'] / 30  # 归一化回合数
                distance_factor = (
                    feature['p1_distance_run'] + feature['p2_distance_run']) / 122  # 归一化跑动距离
                momentum_score += 2.0 * rally_factor * distance_factor * P_t

                # 多球得分
                if feature['rally_count'] > 10:
                    momentum_score += 0.5

                # ACE 得分
                if (player_number == 1 and feature['p1_ace'] > 0) or (player_number == 2 and feature['p2_ace'] > 0):
                    momentum_score += 0.02

                # # 网前得分
                # if (player_number == 1 and feature['p1_net_pt_won'] > 0) or (player_number == 2 and feature['p1_net_pt_won'] > 0):
                #     momentum_score += 0.5

                # 非强迫性失误
                if (player_number == 1 and feature['p1_unf_err'] > 0) or (player_number == 2 and feature['p2_unf_err'] > 0):
                    momentum_score -= 0.05

            momentum_scores[i] = momentum_score

        return momentum_scores

    def generate_data_evilx(self, if_match_id=False) -> pd.DataFrame:

        x1_ls, x2_ls, x3_ls, x4_ls, x5_ls, x6_ls, x7_ls, x8_ls, x9_ls, x10_ls, x11_ls = [
        ], [], [], [], [], [], [], [], [], [], []

        label_ls = []
        match_ls = []

        for match_id, set_no, game_no, point_no in zip(self.df.match_id, self.df.set_no, self.df.game_no, self.df.point_no):
            match = self.df[self.df.match_id == match_id]
            set_ = match[match.set_no == set_no]
            game_ = set_[set_.game_no == game_no]
            point_ = game_[game_.point_no == point_no]
            # 本场 game 的得分领先进度
            x1 = point_['p1_score'].values[0] - point_['p2_score'].values[0]
            # 上一个 point是否得分
            x2 = 0 if x1 < 0 else 1
            # 是否发球得分（无触碰）
            x3 = 1 if 1 in game_['p1_ace'].values else 0
            # 是否回击得分（无触碰）
            x4 = 1 if 1 in game_['p1_winner'].values else 0
            # 本场 game 是否出现双误
            x5 = 1 if 1 in game_['p1_double_fault'].values else 0
            # 本场 game 是否出现非强迫失误
            x6 = 1 if 1 in game_['p1_unf_err'].values else 0
            # 上网次数与上网得分比例
            x7 = game_['p1_net_pt_won'].sum(
            )/game_['p1_net_pt'].sum() if game_['p1_net_pt'].sum() != 0 else 0
            # 最近三个 point 内的总计跑图里程
            x8 = point_['p1_distance_run'].values[0]
            # 接发得分率
            x9 = 1 if 1 in game_['server'].values else 0
            # 是否为发球局
            x10 = 1 if point_['serve_no'].values[0] == 1 else 0

            label = 1 if point_['point_victor'].values[0] == 1 else 0
            label_ls.append(label)

            x1_ls.append(x1)
            x2_ls.append(x2)
            x3_ls.append(x3)
            x4_ls.append(x4)
            x5_ls.append(x5)
            x6_ls.append(x6)
            x7_ls.append(x7)
            x8_ls.append(x8)
            x9_ls.append(x9)
            x10_ls.append(x10)

            data_dir = self.get_new_folder_name()

            if if_match_id:

                match_id = point_['match_id'].values[0]
                match_ls.append(match_id)

                new_file_name = 'Standard_Training_Data_' + \
                    self.get_origin_file_name().split('.')[0] + '_match_id.csv'
                new_file_name = os.path.join(data_dir, new_file_name)

                dataset = pd.DataFrame({'x1': x1_ls, 'x2': x2_ls, 'x3': x3_ls, 'x4': x4_ls,
                                        'x5': x5_ls, 'x6': x6_ls, 'x7': x7_ls, 'x8': x8_ls, 'x9': x9_ls, 'x10': x10_ls, 'label': label_ls, 'match_id': match_ls})
            else:
                new_file_name = 'Standard_Training_Data_' + \
                    self.get_origin_file_name().split('.')[0] + '.csv'
                new_file_name = os.path.join(data_dir, new_file_name)

                dataset = pd.DataFrame({'x1': x1_ls, 'x2': x2_ls, 'x3': x3_ls, 'x4': x4_ls,
                                        'x5': x5_ls, 'x6': x6_ls, 'x7': x7_ls, 'x8': x8_ls, 'x9': x9_ls, 'x10': x10_ls, 'label': label_ls})

        # 计算整体势头得分
        comprehensive_momentum_1 = self.calculate_comprehensive_momentum(1)
        # comprehensive_momentum_2 = self.calculate_comprehensive_momentum(2)
        # 将返回的势头得分添加到数据集中的倒数第二列
        dataset.insert(10, 'x11', comprehensive_momentum_1)

        dataset.to_csv(new_file_name, index=False)

        return dataset

    def find_player_in_match(self) -> dict:

        players_matches = {}

        for index, row in self.df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            match_id = row['match_id']

            players_matches.setdefault(player1, set()).add(match_id)
            players_matches.setdefault(player2, set()).add(match_id)

        players_matches = {k: list(v) for k, v in players_matches.items()}

        data_dir = self.get_new_folder_name()
        json_file_name = 'player_in_match_' + self.get_origin_file_name() + '.json'
        json_file_name = os.path.join(data_dir, json_file_name)

        with open(json_file_name, 'w') as json_file:
            json.dump(players_matches, json_file, ensure_ascii=False, indent=2)

        print('json_file_name:', json_file_name)

        return players_matches

    def calculate_turning_points(self, window_size=4) -> list:

        data_dir = self.get_new_folder_name()

        new_file_name_with_turning_points = 'Standard_Training_Data_' + \
            self.get_origin_file_name().split('.')[0] + '_turning_points.csv'
        new_file_name_with_turning_points = data_dir + \
            '/' + new_file_name_with_turning_points

        turning_points = calculate_turning_points(self.generate_data_evilx(
            if_match_id=True), window_size, new_file_name_with_turning_points)

        print('new_file_name_with_turning_points:',
              new_file_name_with_turning_points)
        return turning_points

    # 对完整比赛进行拐点检测，将预测结果保存到列表中
    def match_turning_point_pred(self) -> list:

        data_dir = self.get_new_folder_name()

        new_file_name_pred = data_dir + '/' + 'Standard_Training_Data_' + \
            self.get_origin_file_name().split('.')[0] + '_match_id_pred.csv'
        feature_importance_file_name = data_dir + '/' + 'feature_importance_' + \
            self.get_origin_file_name().split('.')[0] + '.csv'
        model_performance_file_name = data_dir + '/' + 'model_rf_performance_' + \
            self.get_origin_file_name().split('.')[0] + '.json'

        turning_points_pred = match_turning_point_pred(self.calculate_turning_points(
        )[1], new_file_name_pred, feature_importance_file_name, model_performance_file_name)

        print('new_file_name_pred:', new_file_name_pred)
        print('feature_importance_file_name:', feature_importance_file_name)
        print('model_performance_file_name:', model_performance_file_name)
        return turning_points_pred

    # 按照比赛进行随机森林特征重要性检测，每场比赛分别计算特征重要性得分
    def feature_importance_divided_by_match_id(self):

        data = self.calculate_turning_points()[1]
        list_match = data['match_id'].unique()

        data_dir = self.get_new_folder_name()

        for match in list_match:
            match_data = data[data['match_id'] == match]
            # 保存到 csv 文件中，包含 'Feature Name', 'Feature Importance', 'match_id' 三列
            feature_importances_df = feature_importance_calculation(match_data)

            new_file_name = 'feature_importance_' + match + '.csv'
            new_file_name = os.path.join(data_dir, new_file_name)

            feature_importances_df.to_csv(new_file_name, index=False)

            print(
                f"Feature importance for match {match} saved to {new_file_name}")

    def win_predict_devide(self):

        data = self.calculate_turning_points()[1]

        win_predict_df = win_predict_LR_model(data)

        data_dir = self.get_new_folder_name()

        new_file_name = 'win_predict_' + \
            self.get_origin_file_name().split('.')[0] + '.csv'
        new_file_name = os.path.join(data_dir, new_file_name)

        win_predict_df.to_csv(new_file_name, index=False)

        print(f"Win prediction saved to {new_file_name}")

    def match_performance_calc(self):

        data = self.calculate_turning_points()[1]

        data_dir = self.get_new_folder_name()

        new_file_name_pred = data_dir + '/' + 'match_performance_' + \
            self.get_origin_file_name().split('.')[0] + '.csv'
        feature_importance_file_name = data_dir + '/' + 'feature_importance_' + \
            self.get_origin_file_name().split('.')[0] + '.csv'
        model_performance_file_name = data_dir + '/' + 'model_lr_performance_' + \
            self.get_origin_file_name().split('.')[0] + '.json'

        match_performance_calc(
            data, new_file_name_pred, feature_importance_file_name, model_performance_file_name)

        print('new_file_name_pred:', new_file_name_pred)
        print('feature_importance_file_name:', feature_importance_file_name)
        print('model_performance_file_name:', model_performance_file_name)

        return new_file_name_pred, feature_importance_file_name, model_performance_file_name

# # 读取config.json文件
# with open('config.json', 'r') as config_file:
#     config_data = json.load(config_file)

# df_path = config_data["origin_data_path"]
# data = DataCleanAndGenerate(df_path)

# # df_path = '2024_MCM-ICM_Problems/Wimbledon_featured_matches.csv'
# # data = DataCleanAndGenerate(df_path)

# data.generate_data_evilx(if_match_id=True)
# data.find_player_in_match()
# data.calculate_turning_points()
# data.match_turning_point_pred()
# data.feature_importance_divided_by_match_id()
# # data.win_predict_devide()
# data.match_performance_calc()
