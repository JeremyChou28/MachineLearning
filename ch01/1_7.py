# @description:numpy库的使用
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/9/21 15:42

from numpy import *     #导入numpy库
print(random.rand(4, 4))    #生成四阶的随机数组
randMat = mat(random.rand(4, 4))    #生成四阶的随机矩阵
print(randMat.I)        #矩阵求逆
invRandMat = randMat.I
myEye = randMat * invRandMat    #单位矩阵
print(myEye - eye(4))
