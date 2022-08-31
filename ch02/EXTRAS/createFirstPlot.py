# @description:
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/9/21 15:42
from numpy import *
import KNN
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat,datingLabels = KNN.file2matrix('E:\Python_language\MachineLearningProjects\ch02\datingTestSet.txt')
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
ax.axis([-2, 25, -0.2, 2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()
