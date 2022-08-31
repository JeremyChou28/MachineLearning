# @description:从疝气病症预测病马的死亡率 数据集网站：http://archive.ics.uci.edu/ml/datasets/Horse+Colic
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/4 14:14
# 收集数据：给定数据文件
# 准备数据：用python解析文本文件并填充缺失值
# 分析数据：可视化并观察数据
# 训练算法：使用优化算法，找到最佳的回归系数
# 测试算法：为了量化回归的效果，需要观察错误率。根据错误率决定是否回退到训练阶段，通过迭代的次数和步长等参数来得到
# 更好的回归系数
# 使用算法：实现一个简单的命令行程序来收集马的症状并输出预测结果
import logRegres
from numpy import *


# Logistic回归分类函数
def classify_vector( inx , weights ):
    prob = logRegres.sigmoid( sum( inx * weights ) )

    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test( ):
    frtrain = open( 'horseColicTraining.txt' );
    frtest = open( 'horseColicTest.txt' )
    train_set = []
    train_labels = []

    for line in frtrain.readlines( ):
        curr_line = line.strip( ).split( '\t' )
        line_array = []
        for i in range( 21 ):
            line_array.append( float( curr_line[i] ) )
        train_set.append( line_array )
        train_labels.append( float( curr_line[21] ) )

    train_weights = logRegres.stoc_grad_ascent1( array( train_set ) , train_labels , 1000 )
    error_count = 0
    num_test_vector = 0.0

    for line in frtest.readlines( ):
        num_test_vector += 1.0
        curr_line = line.strip( ).split( '\t' )
        line_array = []
        for i in range( 21 ):
            line_array.append( float( curr_line[i] ) )
        if int( classify_vector( array( line_array ) , train_weights ) ) != int( curr_line[21] ):
            error_count += 1

    error_rate = (float( error_count ) / num_test_vector)
    print( f"the error rate of this test is: {error_rate}" )

    return error_rate


# 多重测试
def mul_test( ):
    num_tests = 10
    error_sum = 0.0

    for k in range( num_tests ):
        error_sum += colic_test( )

    print( f"after {num_tests} iterations the average error rate is: {error_sum / float( num_tests )}" )


mul_test( )
