# @description:KNN对鸢尾花分类
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/9/19 10:35
import numpy as np
import pandas as pd


class KNN:
    """使用python实现K近邻算法"""

    def __init__( self , k ):
        """初始化方法

        Parameters:
        ----
        k:int
            邻居的个数
        """
        self.k = k

    def fit( self , X , y ):
        """训练方法
        Parameters
        ----
            X：类似数组类型，list，ndarray……形状：[样本的数量，特征的数量]
            y：类似数组类型，形状为[样本数量]
                每个样本的目标值，也是就是标签
        """
        # 将X转换成ndarray类型，如果X已经是ndarray则不进行转换
        self.X = np.asarray( X )
        self.y = np.asarray( y )

    def predict( self , X ):
        """根据参数传递的样本，对样本数据进行预测，返回预测之后的结果
        Parameters
        ----
        X：类似数组类型，list，ndarray……形状：[样本的数量，特征的数量]

        Return
        ----
        result：数类型，预测的结果。
        """
        print( data )
        # 将测试的X转换为ndarray结构
        X = np.asarray( X )
        result = []

        for x in X:
            # ndarray相减为对应元素相减，测试的X的每一行与self.X 相减
            # 求欧氏距离：每个元素都取平方值
            dis = np.sqrt( np.sum( (x - self.X) ** 2 , axis=1 ) )
            # 求最近的k个点的距离，sort()排序不适用，因为排序后打乱了顺序
            # argsort()，返回每个元素在排序之前原数组的索引
            index = dis.argsort( )
            # 取前k个元素,距离最近的k的元素
            index = index[:self.k]
            # 返回数组中每个元素出现的次数，元素必须是非负整数
            count = np.bincount( self.y[index] )
            # 返回ndarray之最大的元素的索引，该索引就是我们判定的类别
            result.append( count.argmax( ) )
        return np.asarray( result )

    def score( self , r , Y ):
        return r / Y.size


# 读取数据集，header参数来指定参数标题的行，默认为0，第一行，如果没有标题使用None
data = pd.read_csv( 'iris.csv' , header=0 )
# 对文本进行处理，将Species列的文本映射成数值类型
data['Species'] = data['Species'].map( { 'virginica': 0 , 'setosa': 1 , 'versicolor': 2 } )
# data.head(20)
# 显示末尾行数
# data.tail(20)
# 随机显示，默认为1条
data.sample( 10 )
# 删除不需要的列
data.drop( "id" , axis=1 , inplace=True )
# 重复值检查，any()，一旦有重复值，就返回True
data.duplicated( ).any( )
# 删除重复的数据
data.drop_duplicates( inplace=True )
# 查看各类别的数据条数
# print(data['Species'].value_counts())
# print(data)

# 数据集拆分成训练集和测试集
# 1、提取每个类别鸢尾花的数量
t0 = data[data['Species'] == 0]
t1 = data[data['Species'] == 1]
t2 = data[data['Species'] == 2]

# 打乱顺序，random_state ,记住打乱的顺序
t0 = t0.sample( len( t0 ) , random_state=0 )
t1 = t1.sample( len( t1 ) , random_state=0 )
t2 = t2.sample( len( t2 ) , random_state=0 )
train_X = pd.concat( [t0.iloc[:40 , :-1] , t1.iloc[:40 , :-1] , t2.iloc[:40 , :-1]] , axis=0 )
train_Y = pd.concat( [t0.iloc[:40 , -1] , t1.iloc[:40 , -1] , t2.iloc[:40 , -1]] , axis=0 )
test_X = pd.concat( [t0.iloc[40: , :-1] , t1.iloc[40: , :-1] , t2.iloc[40: , :-1]] , axis=0 )
test_Y = pd.concat( [t0.iloc[40: , -1] , t1.iloc[40: , -1] , t2.iloc[40: , -1]] , axis=0 )
# print(train_X.shape)
# print(train_Y.shape)
# print(test_X.shape)
# print(test_Y.shape)
# 进行训练与测试
knn = KNN( k=3 )
# 进行训练
knn.fit( train_X , train_Y )
# 进行测试
result = knn.predict( test_X )
# display(result)
# display(test_Y)
# 查看预测结果
# display(np.sum(result == test_Y))
r = np.sum( result == test_Y )
print( "测试集的正确率：{:.2f}".format( knn.score( r , test_Y ) ) )
