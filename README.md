# MCM 2024

美赛怎么办吧。

TODO:

- [ ] （Q1）寻找机器学习模型要求效果比LR好
- [ ] （Q1）在`xi`这10个指标中的，使用SPSS找出显著5个指标，再训练出更好的模型
- [x] （Q2）LR_v1 输出内容的解释
- [ ] 优化输出图片样式

## 代码说明

目录结构：

```text
.
├── LR_v1.py
├── LR_v2.py
├── data_cleanup_and_generate.py
├── plot_part.py
├── random_forest_model_setup.py
├── utils
│   ├── data_set_turning_point_v1.py
│   ├── data_set_turning_point_v2.py
│   └── momentum.py
├── win_predict_v1.py
└── win_predict_v2.py
```

运行顺序：

1. `data_cleanup_and_generate.py`: 数据清洗和生成，用于输出各种数据，必须运行。

2. `LR_vi.py`: 逻辑回归模型，仅用于模型分析，可以不运行。

   1. v1 是用于通过`xi`等变量预测「该名选手是否获胜」的模型
   2. v2 是用于通过原始数据中的部分参数预测「该名选手是否获胜」的模型

3. `random_forest_model_setup.py`: 随机森林模型，根据`xi`等变量预测「动量是否出现拐点」，可以不运行。在`data_cleanup_and_generate.py`中用于生成数据。
4. `win_predict_vi.py`: 用于预测「该名选手是否获胜」，可以不运行。

   1. v1 是包含2名选手的数据，但是计算方法有误
   2. v2 是包含1名选手的数据，比 v1 真实性更高，但是计算方法也有误

5. `plot_part.py`: 用于绘制图表，可以不运行。
