import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('data/Standard_Training_Data_match_id.csv')

# 获取x11列的数据
x11 = data['x11']

# 检测转折点，使用 np.diff() 函数
turning_points = []
dx11 = x11.diff()
for i in range(1, len(dx11) - 1):
    if dx11[i] * dx11[i - 1] < 0:
        turning_points.append(i)

# 输出转折点
print("Turning Points", turning_points)
print("转折点个数：", len(turning_points))

data_with_turning_points = data.copy()

# 添加转折点标记，在转折点处标记为1，新的列名为turning_point
data_with_turning_points['turning_point'] = 0
for i in turning_points:
    data_with_turning_points.loc[i, 'turning_point'] = 1
# 保存数据
data_with_turning_points.to_csv('data/Standard_Training_Data_match_id_with_turning_points.csv', index=False)

# 绘制图表
plt.plot(x11)
plt.scatter(turning_points, [x11[i] for i in turning_points], color='red')
plt.xlabel('time')
plt.ylabel('x11 value')
plt.title('Turning Points')
plt.show()
