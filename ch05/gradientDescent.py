# @description:梯度下降优化算法
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/4 11:51
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# 加载数据
def load_dataset( n ):
    noise = np.random.rand( n )
    X = [[x , 1.] for x in range( n )]
    y = [(0.5 * X[i][0] + 1. + noise[i]) for i in range( n )]
    return np.array( X ).T , np.array( y ).T  # 注意X，W，y的维数


def linearRegLsq( x , y ):
    '''
    最小二乘法直接求解theta
    :param x:
    :param y:
    :return:
    '''
    xtx = np.dot( x.T , x )
    if np.linalg.det( xtx ) == 0.0:  # 判断xtx行列式是否等于0，奇异矩阵不能求逆
        print( 'Can not resolve the problem' )
        return
    theta_lsq = np.dot( np.dot( np.linalg.inv( np.dot( x.T , x ) ) , x.T ) , y )
    return theta_lsq


def linearRegBgd1( x , y , alpha=0.005 , lamba=0.005 , loop_max=1000 ):
    '''
    批量梯度下降算法  公式：  theta=theta + alpha * sum( (y-y_hat) * x)
    :param x:
    :param y:
    :param alpha: alpha = 0.005       #学习率步长，learning rate
    :param lamba: lamba=0.005     #惩罚系数
    :param loop_max: 迭代次数
    :return:
    '''

    np.set_printoptions( precision=4 )  # 设置numpy输出4位小数
    n = len( x )  # 取第一行数据，查看数据列数
    theta = np.zeros( n )  # 初始化theta数组，默认全为0
    for times in range( loop_max ):
        y_hat = np.dot( x.T , theta )
        loss = y - y_hat
        for i in range( 1 , n ):
            if i == 1:
                loss_n = np.row_stack( (loss , loss) )
            elif i > 1:
                loss_n = np.row_stack( (loss_n , loss) )

        theta_old = theta
        theta = theta + (alpha * x.T * loss_n.T).sum( axis=0 )  # +lamba*theta
        if (theta - theta_old).all( ) < 0.001:
            break
        # print('x:',x,'y:',y,'loss:',loss)
        # print('times:',times,'theta:',theta)
        # print('')
    return theta


def linearRegBgd2( x , y , alpha=0.005 , lamba=0.005 , loop_max=1000 ):
    '''
    批量梯度下降算法  公式：  theta=theta + alpha * sum( (y-y_hat) * x)
    :param x:
    :param y:
    :param alpha:alpha = 0.005   #学习率步长，learning rate
    :param lamba:lamba=0.005     #惩罚系数
    :param loop_max:
    :return:
    '''

    np.set_printoptions( precision=4 )  # 设置numpy输出4位小数
    n = len( x[0] )  # 取第一行数据，查看数据列数
    theta = np.zeros( n )  # 初始化theta数组，默认全为0

    for times in range( loop_max ):
        y_hat = np.dot( x , theta.T ).reshape( (-1 , 1) )
        loss = (y_hat - y)  # 此处是y_hat-y，对应的theta求解处增加了一个负号
        loss_n = np.array( [] )  # 为方便直接用矩阵实现sum( (y-y_hat) * x)，产生多列loss乘以loss后求和
        if n == 1:  # 判断x矩阵有几列，列为1的时候单独处理
            loss_n = loss
        else:
            for i in range( 1 , n ):  # 判断x矩阵有几列，列为大于2的时候，产生n列loss
                # print(i)
                if i == 1:
                    loss_n = np.column_stack( (loss , loss) )
                elif i > 1:
                    loss_n = np.column_stack( (loss_n , loss) )

        theta_old = theta  # 记录迭代前的theta
        # tmp=alpha*(loss_n*x).sum(axis=0)

        theta = theta - (alpha * (x * loss_n)).sum( axis=0 )  # +lamba*theta
        # 使用矩阵求解theta，注意此处的负号，loss*x后按特征列求和作为theta，注意上方本来想实现惩罚项的，但没有想明白怎么实现

        if (theta - theta_old).all( ) < 0.001:  # 判断前后两次theta变换情况，当变化率很小时跳出循环
            break
        # print('x:',x,'y:',y,'loss:',loss)
        # print('y_hat:',y_hat.T)
        # print('times:',times,'theta:',theta)
        # print('')
    return theta


def linearRegSgd( x , y , alpha=0.005 , lamba=0.005 , loop_max=10000 ):
    '''
    随机梯度下降算法  公式：for i=1 to m:
        theta=theta + alpha * (y-y_hat) * x
    :param x:
    :param y:
    :param alpha:alpha = 0.005   #学习率步长，learning rate
    :param lamba:lamba=0.005     #惩罚系数
    :param loop_max:
    :return:
    '''

    np.set_printoptions( precision=4 )  # 设置numpy输出4位小数
    n = len( x[0] )  # 取第一行数据，查看数据列数
    theta = np.zeros( n )  # 初始化theta数组，默认全为0

    for times in range( loop_max ):
        for i in range( 0 , len( x ) ):  # 取其中一条数据进行梯度下降
            # i=0
            y_hat = np.dot( x[i] , theta.T ).reshape( (-1 , 1) )
            loss = (y_hat - y[i])  # 此处是y_hat-y，对应的theta求解处增加了一个负号
            theta_old = theta  # 记录迭代前的theta
            theta = theta - alpha * x[i] * loss[0]  # +lamba*theta
            # 求解theta，注意此处的负号，注意上方本来想实现惩罚项的，但没有想明白怎么实现
            if (theta - theta_old).all( ) < 0.001:  # 判断前后两次theta变换情况，当变化率很小时跳出循环
                break
    #            print('x:',x,'y:',y,'loss:',loss)
    #            print('y_hat:',y_hat.T)
    #            print('times:',times,'theta:',theta)
    #            print('')
    return theta


def linearRegMgd( x , y , alpha=0.005 , lamba=0.005 , loop_max=1000 , batch_size=9 ):
    # alpha = 0.0000001      #学习率步长，learning rate
    # lamba=0.005     #惩罚系数
    '''
    mini-batch梯度下降算法公式：
    每次使用batch_size个数据进行计算
    for i=1 to m:
        theta=theta + alpha * (y-y_hat) * x
    '''
    np.set_printoptions( precision=4 )  # 设置numpy输出4位小数
    n = len( x[0] )  # 取第一行数据，查看数据列数
    theta = np.zeros( n )  # 初始化theta数组，默认全为0

    for times in range( loop_max ):
        for i in range( 0 , int( len( x ) / batch_size ) + 1 ):
            # print('i:',i,x[i*batch_size:(i+1)*batch_size])
            x_mini = x[i * batch_size:(i + 1) * batch_size]
            y_mini = y[i * batch_size:(i + 1) * batch_size]

            y_hat = np.dot( x_mini , theta.T ).reshape( (-1 , 1) )
            loss = (y_hat - y_mini)  # 此处是y_hat-y，对应的theta求解处增加了一个负号
            loss_n = np.array( [] )  # 为方便直接用矩阵实现sum( (y-y_hat) * x)，产生多列loss乘以loss后求和
            if n == 1:  # 判断x矩阵有几列，列为1的时候单独处理
                loss_n = loss
            else:
                for i in range( 1 , n ):  # 判断x矩阵有几列，列为大于2的时候，产生n列loss
                    # print(i)
                    if i == 1:
                        loss_n = np.column_stack( (loss , loss) )
                    elif i > 1:
                        loss_n = np.column_stack( (loss_n , loss) )

            theta_old = theta  # 记录迭代前的theta
            # tmp=alpha*(loss_n*x).sum(axis=0)

            theta = theta - (alpha * (x_mini * loss_n)).sum( axis=0 )  # +lamba*theta
            # 使用矩阵求解theta，注意此处的负号，loss*x后按特征列求和作为theta，注意上方本来想实现惩罚项的，但没有想明白怎么实现

            if (theta - theta_old).all( ) < 0.001:  # 判断前后两次theta变换情况，当变化率很小时跳出循环
                break

        # print('x:',x,'y:',y,'loss:',loss)
        # print('y_hat:',y_hat.T)
        # print('times:',times,'theta:',theta)
        # print('')
    return theta


if __name__ == "__main__":
    path = 'E:\Python_language\MachineLearningProjects\ch05\Advertising.csv'
    data = pd.read_csv( path )  # TV、Radio、Newspaper、Sales
    # data_matrix = data.as_matrix(columns=['TV', 'Radio', 'Newspaper', 'Sales'])  # 转换数据类型
    data_array = data.values[: , 1:]  # 转换数据类型，去除第一列序号列
    x = data_array[: , :-1]  # 去除y对应列的数据，其他列作为x
    y = data_array[: , -1].reshape( (-1 , 1) )

    # 0、绘制图像，查看数据分布情况
    plt.plot( data['TV'] , y , 'ro' , label='TV' )
    plt.plot( data['radio'] , y , 'g^' , label='Radio' )
    plt.plot( data['newspaper'] , y , 'mv' , label='Newspaer' )
    plt.legend( loc='lower right' )
    plt.grid( )
    plt.show( )

    # 1、使用sklearn LinearRegression包求解θ
    linreg = LinearRegression( )
    model = linreg.fit( x , y )
    print( '' )
    print( '1、sklearn LinearRegression包求解θ:' , 'coef:' , linreg.coef_[0] , ',intercept:' , linreg.intercept_ )

    # 2、最小二乘法，直接求解析解θ
    theta_lsq = linearRegLsq( x , y )
    print( '' )
    # 报错：cannot reshape array of size 2 into shape (1,3)
    # 这是因为保存的CSV文件中表头对应不对
    print( '2、最小二乘法，theta解析解：' , theta_lsq.reshape( 1 , 3 )[0] )

    # 3、批量梯度下降求解theta
    # 注意下面两个函数alpha都是非常小的值，取过大的值时，不收敛
    # x1, y1 = load_dataset(10)
    # theta1=linearRegBgd1(x1, y1)

    theta1 = linearRegBgd1( x.T , y.T , alpha=0.0000001 )
    print( '' )
    print( '3.1、批量梯度下降，linearRegBgd1函数，theta：' , theta1 )

    theta2 = linearRegBgd2( x , y , alpha=0.0000001 )
    print( '' )
    print( '3.2、批量梯度下降，linearRegBgd2函数，theta：' , theta2 )

    theta3 = linearRegSgd( x , y , alpha=0.000001 , loop_max=10000 )
    print( '' )
    print( '3.3、随机梯度下降，linearRegSgd函数，theta：' , theta3 )

    theta4 = linearRegMgd( x , y , alpha=0.000001 , loop_max=10000 , batch_size=11 )
    print( '' )
    print( '3.4、mini-batch梯度下降，linearRegMgd函数，theta：' , theta4 )
