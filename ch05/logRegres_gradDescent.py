# @description:
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/4 14:04
import matplotlib.pyplot as plt
from numpy import *
from numpy.ma import array


# 加载数据
def load_data_set( ):
    '''
    加载数据
    :return:数据集中输入的属性数据X0,X1,X2；标签数组
    '''
    data_matrix = []
    label_matrix = []

    fr = open( 'testSet.txt' )

    for line in fr.readlines( ):
        line_array = line.strip( ).split( )
        data_matrix.append( [1.0 , float( line_array[0] ) , float( line_array[1] )] )  # 设置输入的X0为1
        label_matrix.append( int( line_array[2] ) )  # 加入标签

    return data_matrix , label_matrix


def sigmoid( inx ):
    if inx.all( ) >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + exp( -inx ))
    else:
        return exp( inx ) / (1 + exp( inx ))


# Logistic回归梯度上升优化算法
def grad_ascent( data_matrix , class_label ):
    # 将数据转化为numpy矩阵
    data_matrix = mat( data_matrix )
    label_matrix = mat( class_label ).transpose( )
    m , n = shape( data_matrix )  # 得到数据矩阵的大小

    alpha = 0.01  # 设置步长
    max_cycles = 500  # 设置迭代次数
    weights = ones( (n , 1) )  # 初始化回归系数矩阵

    # 矩阵之间做乘法
    for k in range( max_cycles ):
        h = sigmoid( data_matrix * weights )
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose( ) * error

    return weights


dataArr , labelMat = load_data_set( )
weights = grad_ascent( dataArr , labelMat )

print( dataArr )
print( labelMat )
print( weights )


# 画图
def plot_best_fit( weights ):
    data_matrix , label_matrix = load_data_set( )
    data_array = array( data_matrix )
    n = shape( data_array )[0]
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range( n ):
        if int( label_matrix[i] ) == 1:
            x1.append( data_array[i , 1] )
            x2.append( data_array[i , 2] )
        else:
            y1.append( data_array[i , 1] )
            y2.append( data_array[i , 2] )

    fig = plt.figure( )
    ax = fig.add_subplot( 111 )

    ax.scatter( x1 , x2 , s=30 , c='red' , marker='s' , label='1' )
    ax.scatter( y1 , y2 , s=30 , c='green' , label='0' )
    x = arange( -3.0 , 3.0 , 0.1 )

    # 最佳似合直线
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot( x , y )
    ax.legend( )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.show( )


# weights = grad_ascent( dataArr , labelMat )
#
# plot_best_fit( weights.getA( ) )


# 随机梯度上升算法
def stroc_grad_ascent0( data_matrix , class_labels ):
    m , n = shape( data_matrix )
    alpha = 0.01
    weights = ones( n )

    for i in range( m ):
        h = sigmoid( sum( data_matrix[i] * weights ) )
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]

    return weights


#
# weights = stroc_grad_ascent0( array( dataArr ) , labelMat )
# plot_best_fit( weights )


# 改进后的随机梯度上升算法
def stoc_grad_ascent1( data_matrix , class_labels , num_iter=150 ):
    m , n = shape( data_matrix )
    weights = ones( n )

    for j in range( num_iter ):
        data_index = list( range( m ) )
        for i in range( m ):
            # 每次来调整alpha的值
            alpha = 4 / (1.0 + j + i) + 0.0001

            # 随机选取更新
            rand_index = int( random.uniform( 0 , len( data_index ) ) )
            h = sigmoid( sum( data_matrix[rand_index] * weights ) )
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del (data_index[rand_index])

    return weights

# weights = stoc_grad_ascent1( array( dataArr ) , labelMat )
# plot_best_fit( weights )
