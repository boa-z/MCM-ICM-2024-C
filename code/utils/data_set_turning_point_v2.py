import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

window = 5 # 窗口大小

# 计算窗口内的趋势
def calculate_trend(window):
    return np.polyfit(np.arange(len(window)), window, 1)[0]

# 编写一个函数，用于计算转折点，并输出转折点的位置到csv文件中
def calculate_turning_points(data, window_size=5, new_file_name_with_turning_points = 'data/XXX_with_turning_points.csv'):
    # 获取x11列的数据
    x11 = data['x11']
    # 转化为numpy数组
    x11 = np.array(x11, dtype=float)
    
    turning_points = []
    # 定义阈值
    threshold = 10
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        window = window[window.apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull()]  # Remove non-numeric values
        window = np.array(window, dtype=float)  # Convert window to float type
        trend = calculate_trend(window)

        # 判断趋势是否发生变化
        if i > 0 and np.any(np.abs(trend - previous_trend) > threshold):
            turning_points.append(i)

        previous_trend = trend

    # 先计算所有比赛的转折点，保存到列表中
    data_with_turning_points = data.copy()
    # 添加转折点标记，在转折点处标记为1，新的列名为turning_point
    data_with_turning_points['turning_point'] = 0
    for i in turning_points:
        data_with_turning_points.loc[i, 'turning_point'] = 1
    # 保存数据
    data_with_turning_points.to_csv(new_file_name_with_turning_points, index=False)
    
    # 输出转折点
    print("转折点计算完成，转折点个数：", len(turning_points))
    return turning_points, data_with_turning_points


# # 读取数据
# data = pd.read_csv('data/Standard_Training_Data_Wimbledon_featured_matches_match_id.csv')
# calculate_turning_points(data, window_size=window, new_file_name_with_turning_points = 'data/Standard_Training_Data_Wimbledon_featured_matches_match_id_with_turning_points.csv')
