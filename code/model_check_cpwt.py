'''
参考自九条代码之多个模型检查的代码
'''

'''
基于九条之代码，补充一些要计算的数值
生成数据后，请使用 SPSS 进行数据分析
'''

# 剔除了得分异常的样本,删除了配速有缺失的样本

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
import seaborn as sns

# 读取数据

dataset = pd.read_csv('data/Wimbledon_featured_matches/Standard_Training_Data_Wimbledon_featured_matches_match_id.csv')

scaler = MinMaxScaler()
columns = dataset.columns[:-3]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)

import numpy as np
from sklearn import metrics

def f(model_list,name_list,types='train'):
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')    # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  #绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    if types == 'test':
        for model,name in zip(model_list,name_list):
            ytest_prob = model.predict_proba(xvalid)[:,1]
            fpr, tpr, _ = metrics.roc_curve(yvalid, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2, label=f'{name} AUC:%.4f' % auc)  # 绘制AUC 曲线
    else:
        for model,name in zip(model_list,name_list):
            ytest_prob = model.predict_proba(xtrain)[:,1]
            fpr, tpr, _ = metrics.roc_curve(ytrain, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2, label=f'{name} AUC:%.4f' % auc)  # 绘制AUC 曲线
    plt.legend(loc='upper left',fontsize=15)    # 设置显示标签的位置
    plt.xlabel('False Positive Rate', fontsize=14)   #绘制x,y 坐标轴对应的标签
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.tick_params(labelsize=23)

    plt.grid(True, ls=':')  # 绘制网格作为底板;b是否显示网格线；ls表示line style
    plt.savefig(f'figure/问题1_roc_auc({types}(采样前)).png',dpi=500)
    # plt.show()
    plt.close()

xtrain, xvalid, ytrain, yvalid = train_test_split(dataset[columns].values,dataset['label'].values,random_state=620,test_size=0.2)

# model1 = LGBMClassifier(random_state=30)
# model2 = XGBClassifier(random_state=50)
# model3 = SVC(probability=True,random_state=50)
# model4 = MLPClassifier(random_state=60)
model5 = LogisticRegression(random_state=50)
# 除了原来代码中的模型，再去测试一下其他模型
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

# model6 = RandomForestClassifier(random_state=50)
# model7 = DecisionTreeClassifier(random_state=50)
# model8 = KNeighborsClassifier()
model9 = GaussianNB()
# model10 = AdaBoostClassifier(random_state=50)
# model11 = GradientBoostingClassifier(random_state=50)
# model12 = BaggingClassifier(random_state=50)
# model13 = ExtraTreesClassifier(random_state=50)
# model14 = LinearDiscriminantAnalysis()
# model15 = QuadraticDiscriminantAnalysis()

# model1.fit(xtrain,ytrain)
# model2.fit(xtrain,ytrain)
# model3.fit(xtrain,ytrain)
# model4.fit(xtrain,ytrain)
model5.fit(xtrain,ytrain)
# model6.fit(xtrain,ytrain)
# model7.fit(xtrain,ytrain)
# model8.fit(xtrain,ytrain)
model9.fit(xtrain,ytrain)
# model10.fit(xtrain,ytrain)
# model11.fit(xtrain,ytrain)
# model12.fit(xtrain,ytrain)
# model13.fit(xtrain,ytrain)
# model14.fit(xtrain,ytrain)
# model15.fit(xtrain,ytrain)


# f([model1,model2,model3,model4,model5],['LGBM','XGB','SVC','MLP','LR'],'test')
# f([model1,model2,model3,model4,model5],['LGBM','XGB','SVC','MLP','LR'],'train')

# f([model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,model11,model12,model13,model14,model15],['LGBM','XGB','SVC','MLP','LR','RF','DT','KNN','NB','Ada','GB','Bag','ET','LDA','QDA'],'test')
# f([model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,model11,model12,model13,model14,model15],['LGBM','XGB','SVC','MLP','LR','RF','DT','KNN','NB','Ada','GB','Bag','ET','LDA','QDA'],'train')

f([model5,model9],['LR','NB'],'test')
f([model5,model9],['LR','NB'],'train')