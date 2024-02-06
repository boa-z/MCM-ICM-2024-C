import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# 加载数据
data_file_path = '2024_MCM-ICM_Problems/Wimbledon_featured_matches.csv'
df = pd.read_csv(data_file_path)

# 根据特定的比赛ID筛选数据，并创建副本以避免SettingWithCopyWarning
# target_match_id = '2023-wimbledon-1302'
# filtered_data = df[df['match_id'] == target_match_id].copy()

filtered_data = df.copy()

# 定义一个计算综合势头得分的函数
from code.utils.data_cleanup_and_generate import calculate_comprehensive_momentum


# 为两位球员计算综合势头得分
filtered_data['comprehensive_momentum_1'] = calculate_comprehensive_momentum(filtered_data, player_number=1)
filtered_data['comprehensive_momentum_2'] = calculate_comprehensive_momentum(filtered_data, player_number=2)

# 将两位球员计算综合势头得分输出到 CSV 文件
filtered_data.to_csv('momentum_scores.csv', index=False)

# Define the threshold for a significant momentum shift
threshold = 10

# Initialize lists to store the points of positive and negative shifts for both players
shifts_player_1 = []
shifts_player_2 = []

# Calculate the momentum change for each point and find shifts
for i in range(1, len(filtered_data)):
    change_1 = filtered_data['comprehensive_momentum_1'].iloc[i] - filtered_data['comprehensive_momentum_1'].iloc[i - 1]
    change_2 = filtered_data['comprehensive_momentum_2'].iloc[i] - filtered_data['comprehensive_momentum_2'].iloc[i - 1]
    
    if abs(change_1) >= threshold:
        shift_type = 'Positive' if change_1 > 0 else 'Negative'
        shifts_player_1.append((i, shift_type))
    if abs(change_2) >= threshold:
        shift_type = 'Positive' if change_2 > 0 else 'Negative'
        shifts_player_2.append((i, shift_type))

# Annotation
for point, shift_type in shifts_player_1:
    set_no = filtered_data['set_no'].iloc[point]
    game_no = filtered_data['game_no'].iloc[point]
    print(f"Player 1 had a {shift_type} shift at point number {point}, during set {set_no}, game {game_no}.")

for point, shift_type in shifts_player_2:
    set_no = filtered_data['set_no'].iloc[point]
    game_no = filtered_data['game_no'].iloc[point]
    print(f"Player 2 had a {shift_type} shift at point number {point}, during set {set_no}, game {game_no}.")
    
# Create the plot
plt.figure(figsize=(12, 6))
# plt.plot(filtered_data['point_no'], filtered_data['comprehensive_momentum_1'], label='Player 1 Momentum', color='blue')
# plt.plot(filtered_data['point_no'], filtered_data['comprehensive_momentum_2'], label='Player 2 Momentum', color='red')

sns.lineplot(x='point_no', y='comprehensive_momentum_1', data=filtered_data, label='Player 1 Momentum', color='green')
sns.lineplot(x='point_no', y='comprehensive_momentum_2', data=filtered_data, label='Player 2 Momentum', color='red')

for point, shift_type in shifts_player_1:
    marker = '*' if shift_type == 'Positive' else 'x'
    color = 'green' if shift_type == 'Positive' else 'red'
    plt.scatter(filtered_data['point_no'].iloc[point], filtered_data['comprehensive_momentum_1'].iloc[point], 
                color=color, marker=marker, s=100)

for point, shift_type in shifts_player_2:
    marker = '*' if shift_type == 'Positive' else 'x'
    color = 'green' if shift_type == 'Positive' else 'red'
    plt.scatter(filtered_data['point_no'].iloc[point], filtered_data['comprehensive_momentum_1'].iloc[point], 
                color=color, marker=marker, s=100)
    
# Add labels and title to the plot
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Advanced Momentum Score with Consecutive Points Comparison Throughout the Match')
plt.legend()
plt.show()
