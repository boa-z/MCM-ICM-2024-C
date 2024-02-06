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
from model_setup_random_forest import match_turning_point_pred
from model_setup_random_forest import feature_importance_calculation
from utils.win_predict_v2 import win_predict_LR_model
from model_setup_LR_v1 import match_performance_calc


class DataCleanAndGenerate:
    def __init__(self, df_path):
        self.df_path = df_path
        self.df = self.data_clean(df_path)

    # Samples with abnormal scores were eliminated
    # samples with missing pace were deleted.
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

    def calculate_comprehensive_momentum(self, player_number, \
                                         window_size=4) -> list:

        data = self.df
        momentum_scores = [0] * len(data)
        consecutive_point_wins = 0  # Track streaks
        consecutive_game_wins = 0  # Track winning streaks
        previous_game_winner = None  # Track the winner of the previous round
        # Base Momentum Score Addition for Break of Serve
        initial_break_point_value = 1  

        for i in range(1, len(data)):
            recent_data = data[max(0, i - window_size):i]
            momentum_score = 0

            for _, feature in recent_data.iterrows():
                # Basic Momentum Score Calculation
                P_t = 1 if feature['point_victor'] == player_number else -1
                S_t = 1.2 if feature['server'] == player_number else 1.0
                base_momentum = P_t * S_t
                momentum_score += base_momentum
                 # Reset break point value
                break_point_value = initial_break_point_value 

                # Continuous score correction (linear)
                if P_t == 1:
                    consecutive_point_wins += 1
                else:
                    consecutive_point_wins = 0  # Reset when losing points
                momentum_score += 0.03 * consecutive_point_wins

                if feature['game_victor']:
                    current_game_winner = feature['game_victor']
                    if current_game_winner == player_number:
                        if current_game_winner == previous_game_winner:
                            consecutive_game_wins += 1
                        else:
                            consecutive_game_wins = 0
                    previous_game_winner = current_game_winner
                    momentum_score += 0.2 * consecutive_game_wins

                if feature['set_victor']:
                    player1_set = feature['p1_sets'] + \
                        1 if feature['set_victor'] == player_number \
                            else feature['p1_sets']
                    player2_set = feature['p2_sets'] + \
                        1 if feature['set_victor'] == player_number \
                            else feature['p2_sets']
                    diff = (player2_set - player1_set) * \
                        (-1 ** player_number)
                    momentum_score += 0.1 * (2 ** diff)

                if feature['game_victor']:
                    score_diff = \
                        abs(feature['p1_games'] - feature['p2_games'])
                    momentum_score += 0.02 * score_diff * P_t

                if feature['p1_break_pt_missed'] == 1 or \
                    feature['p2_break_pt_missed'] == 1:
                    break_point_value -= 0.1

                if feature['p1_break_pt_won'] == 1 \
                    or feature['p2_break_pt_won'] == 1:
                    break_point_value = max(break_point_value, 0.1)
                    momentum_score += break_point_value * P_t

                rally_factor = feature['rally_count'] / 30
                distance_factor = (
                    feature['p1_distance_run'] + \
                        feature['p2_distance_run']) / 122
                momentum_score += 2.0 * rally_factor * distance_factor * P_t

                if feature['rally_count'] > 10:
                    momentum_score += 0.5

                if (player_number == 1 and feature['p1_ace'] > 0) \
                    or (player_number == 2 and feature['p2_ace'] > 0):
                    momentum_score += 0.02

                if (player_number == 1 and feature['p1_unf_err'] > 0) \
                    or (player_number == 2 and feature['p2_unf_err'] > 0):
                    momentum_score -= 0.05

            momentum_scores[i] = momentum_score

        return momentum_scores

    def generate_data_para_x(self, if_match_id=False) -> pd.DataFrame:

        x1_ls, x2_ls, x3_ls, x4_ls, x5_ls, x6_ls, x7_ls, x8_ls, x9_ls, x10_ls = [
        ], [], [], [], [], [], [], [], [], []

        label_ls = []
        match_ls = []

        for match_id, set_no, game_no, point_no in zip(self.df.match_id, \
                    self.df.set_no, self.df.game_no, self.df.point_no):
            match = self.df[self.df.match_id == match_id]
            set_ = match[match.set_no == set_no]
            game_ = set_[set_.game_no == game_no]
            point_ = game_[game_.point_no == point_no]

            # The score of this game leads the progress
            x1 = point_['p1_score'].values[0] - point_['p2_score'].values[0]
            # Whether the previous point scored
            x2 = 0 if x1 < 0 else 1
            # Whether ACE
            x3 = 1 if 1 in game_['p1_ace'].values else 0
            # whether to score
            x4 = 1 if 1 in game_['p1_winner'].values else 0
            # Is there a double fault in this game?
            x5 = 1 if 1 in game_['p1_double_fault'].values else 0
            # Are there any unforced errors in this game?
            x6 = 1 if 1 in game_['p1_unf_err'].values else 0
            # Ratio of Internet times and Internet score
            x7 = game_['p1_net_pt_won'].sum(
            )/game_['p1_net_pt'].sum() if game_['p1_net_pt'].sum() != 0 else 0
            # The total mileage in the last three points
            x8 = point_['p1_distance_run'].values[0]
            # Return score rate
            x9 = 1 if 1 in game_['server'].values else 0
            # Is it a service game?
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

                dataset = pd.DataFrame({'x1': x1_ls, 'x2': x2_ls, 'x3': x3_ls, \
                                        'x4': x4_ls, 'x5': x5_ls, 'x6': x6_ls, \
                                        'x7': x7_ls, 'x8': x8_ls, 'x9': x9_ls, \
                                        'x10': x10_ls, 'label': label_ls, \
                                        'match_id': match_ls})
            else:
                new_file_name = 'Standard_Training_Data_' + \
                    self.get_origin_file_name().split('.')[0] + '.csv'
                new_file_name = os.path.join(data_dir, new_file_name)

                dataset = pd.DataFrame({'x1': x1_ls, 'x2': x2_ls, 'x3': x3_ls, \
                                        'x4': x4_ls, 'x5': x5_ls, 'x6': x6_ls, \
                                        'x7': x7_ls, 'x8': x8_ls, 'x9': x9_ls, \
                                        'x10': x10_ls, 'label': label_ls})

        # Calculate overall momentum score
        comprehensive_momentum_1 = self.calculate_comprehensive_momentum(1)
        dataset.insert(10, 'x11', comprehensive_momentum_1)

        dataset.to_csv(new_file_name, index=False)

        return dataset
