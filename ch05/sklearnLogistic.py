# @description:sklearn实现logistic regression
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/4 13:25
# encoding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 建立sigmoid函数
def sigmoid( x ):
    x = x.astype( float )
    return 1. / (1 + np.exp( -x ))


# 训练模型，采用梯度下降算法
def train( x_train , y_train , num , alpha , m , n ):
    beta = np.ones( n )
    for i in range( num ):
        h = sigmoid( np.dot( x_train , beta ) )  # 计算预测值
        error = h - y_train.T  # 计算预测值与训练集的差值
        delt = alpha * (np.dot( error , x_train )) / m  # 计算参数的梯度变化值
        beta = beta - delt
        # print('error',error)
    return beta


def predict( x_test , beta ):
    y_predict = np.zeros( len( y_test ) ) + 0.5
    s = sigmoid( np.dot( beta , x_test.T ) )
    y_predict[s < 0.34] = 0
    y_predict[s > 0.67] = 1
    return y_predict


def accurancy( y_predict , y_test ):
    acc = 1 - np.sum( np.absolute( y_predict - y_test ) ) / len( y_test )
    return acc


if __name__ == "__main__":
    data = pd.read_csv( 'iris.csv' )
    x = data.iloc[: , 1:5]
    y = data.iloc[: , 5].copy( )
    y.loc[y == 'setosa'] = 0
    y.loc[y == 'versicolor'] = 0.5
    y.loc[y == 'virginica'] = 1
    x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.3 , random_state=15 )
    m , n = np.shape( x_train )
    alpha = 0.01
    beta = train( x_train , y_train , 1000 , alpha , m , n )
    pre = predict( x_test , beta )
    t = np.arange( len( x_test ) )
    plt.figure( )
    p1 = plt.plot( t , pre )
    p2 = plt.plot( t , y_test , label='test' )
    label = ['prediction' , 'true']
    plt.legend( label , loc=1 )
    plt.show( )
    acc = accurancy( pre , y_test )
    print( 'The predicted value is ' , pre )
    print( 'The true value is ' , np.array( y_test ) )
    print( 'The accuracy rate is ' , acc )

# import xlrd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# data = xlrd.open_workbook('gua.xlsx')
# sheet = data.sheet_by_index(0)
# Density = sheet.col_values(6)
# Sugar = sheet.col_values(7)
# Res = sheet.col_values(8)
# # 读取原始数据
# X = np.array([Density, Sugar])
# # y的尺寸为(17,)
# y = np.array(Res)
# X = X.reshape(17,2)
# # 绘制分类数据
# f1 = plt.figure(1)
# plt.title('watermelon_3a')
# plt.xlabel('density')
# plt.ylabel('ratio_sugar')
# # 绘制散点图（x轴为密度，y轴为含糖率）
# plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
# plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
# plt.legend(loc = 'upper right')
# plt.show()
# # 从原始数据中选取一半数据进行训练，另一半数据进行测试
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# # 逻辑回归模型
# log_model = LogisticRegression()
# # 训练逻辑回归模型
# log_model.fit(X_train, y_train)
# # 预测y的值
# y_pred = log_model.predict(X_test)
# # 查看测试结果
# print(metrics.confusion_matrix(y_test, y_pred))
# print(metrics.classification_report(y_test, y_pred))
