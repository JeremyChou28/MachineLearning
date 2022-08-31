# @description:利用SVM进行手写数字识别
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/4 21:20

import svmMLiA
from numpy import *


# 功能：图像矩阵转化为m*1矩阵
# 输入：文件名
# 输出：m*1矩阵
def img2vector( filename ):
    returnVect = zeros( (1 , 1024) )
    fr = open( filename )
    for i in range( 32 ):
        lineStr = fr.readline( )
        for j in range( 32 ):
            returnVect[0 , 32 * i + j] = int( lineStr[j] )
    return returnVect


# 功能：将图像内容导入矩阵
# 输入：一级子目录
# 输出：图像矩阵，图像标签向量
def loadImages( dirName ):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir( dirName )  # dirName文件夹下的文件名列表
    m = len( trainingFileList )  # dirName文件夹下的文件数目
    trainingMat = zeros( (m , 1024) )
    for i in range( m ):
        fileNameStr = trainingFileList[i]  # 文件名
        fileStr = fileNameStr.split( '.' )[0]  # 去掉.txt的文件名
        classNumStr = int( fileStr.split( '_' )[0] )  # 要识别的数字
        if classNumStr == 9:  # 数字9
            hwLabels.append( -1 )
        else:  # 数字1
            hwLabels.append( 1 )
        trainingMat[i , :] = img2vector( '%s/%s' % (dirName , fileNameStr) )
    return trainingMat , hwLabels


def testDigits( kTup=('rbf' , 10) ):
    dataArr , labelArr = loadImages( 'trainingDigits' )
    b , alphas = svmMLiA.smoP( dataArr , labelArr , 200 , 0.0001 , 10000 , kTup )
    datMat = mat( dataArr )
    labelMat = mat( labelArr ).transpose( )
    svInd = nonzero( alphas.A > 0 )[0]  # 支持向量的下标
    sVs = datMat[svInd]  # 支持向量
    labelSV = labelMat[svInd]  # 支持向量的类别标签
    print( "there are {} Support Vectors".format( shape( sVs )[0] ) )
    m , n = shape( datMat )
    errorCount = 0
    for i in range( m ):
        kernelEval = svmMLiA.kernelTrans( sVs , datMat[i , :] , kTup )
        predict = kernelEval.T * multiply( labelSV , alphas[svInd] ) + b  # 得预测值
        if sign( predict ) != sign( labelArr[i] ):
            errorCount += 1
    print( "the training error rate is {}".format( float( errorCount ) / m ) )
    dataArr , labelArr = loadImages( 'testDigits' )
    errorCount = 0
    datMat = mat( dataArr )
    labelMat = mat( labelArr ).transpose( )
    m , n = shape( datMat )
    for i in range( m ):
        kernelEval = svmMLiA.kernelTrans( sVs , datMat[i , :] , kTup )
        predict = kernelEval.T * multiply( labelSV , alphas[svInd] ) + b  # 得验证集预测值
        if sign( predict ) != sign( labelArr[i] ):
            errorCount += 1
    print( "the test error rate is: {}".format( float( errorCount ) / m ) )


testDigits( ('rbf' , 20) )
