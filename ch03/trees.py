# @description:决策树算法
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/9/21 15:42

import math
import operator
import pickle
import treePlotter

def create_dataset():
    '''
    创建数据集
    :return:数据集
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']   #标签
    return dataSet, labels


def calc_shannon_ent(dataSet):
    '''
    计算给定数据集的信息熵
    :param dataSet: 数据集
    :return: 信息熵
    '''
    num_entries = len(dataSet)      #获取数据集中实例的总数
    label_counts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannoent = 0.0

    # 以二为底求对数
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannoent -= prob * math.log(prob, 2)       
    return shannoent

# myDat,labels=create_dataset()
# # print(myDat)
# myDat[0][-1]='maybe'
# print(myDat)
# print(calc_shannon_ent(myDat))

def split_dataset(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征的返回值
    :return:
    '''
    # 创建新的list对象
    retdataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            # print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])
            # print(reducedFeatVec)
            retdataSet.append(reducedFeatVec)

    return retdataSet
# myDat,labels=create_dataset()
# split_dataset(myDat,0,0)

def choose_best(dataSet):
    '''
    选择的最好的数据集划分方式
    :param dataSet: 数据集
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calc_shannon_ent(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    # 创建唯一分类标签
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValis = set(featList)
        newEntropy = 0.0

        # 计划每种划分的信息墒
        for value in uniqueValis:
            subDataSet = split_dataset(dataSet, i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calc_shannon_ent(subDataSet)
            infoGain = baseEntropy - newEntropy

            # 计算最好的增益墒
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i

    return bestFeature      #返回最好的信息增益
# myDat,labels=create_dataset()
# print(choose_best(myDat))
# print(calc_shannon_ent(split_dataset(myDat,0,0)))
# print(calc_shannon_ent(split_dataset(myDat,0,1)))

def majoritycnt(classList):
    '''
    多数表决来决定叶子节点的分别
    :param classList: 标签的类数组
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount


def create_tree(dataSet, labels):
    '''
    创建树的函数
    :param dataSet: 数据集
    :param labels: 标签列表，包含了数据集中所有特征的标签
    :return:
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):

        # 停止分类直至所有类别相等
        return classList[0]
    if len(dataSet[0]) == 1:

        # 停止分割直至没有更多特征
        return majoritycnt(classList)
    bestfaet = choose_best(dataSet)
    bestfaetlabel = labels[bestfaet]
    mytree = {bestfaetlabel:{}}
    del(labels[bestfaet])

    # 得到包含所有属性的列表
    featvalues = [example[bestfaet] for example in dataSet]
    uniquevalues = set(featvalues)
    for value in uniquevalues:
        sublables = labels[:]
        mytree[bestfaetlabel][value] = create_tree(split_dataset(dataSet, bestfaet, value), sublables)

    return mytree
# myDat,labels=create_dataset()
# print(create_tree(myDat,labels))    #输出树的字典

def classify(inputtree, featlabels, testvec):
    '''
    使用决策树执行分类
    :param inputtree:输入的决策树
    :param featlabels:标签
    :param testvec:测试向量
    :return:
    '''
    firststr = list(inputtree.keys())[0]
    seconddict = inputtree[firststr]
    featindex = featlabels.index(firststr)      #将标签字符串更换成索引
    key = testvec[featindex]
    valueoffeat = seconddict[key]
    if isinstance(valueoffeat, dict):
        classlabel = classify(valueoffeat, featlabels, testvec)
    else:
        classlabel = valueoffeat
    return classlabel
# myDat,labels=create_dataset()
# print(labels)
# myTree=treePlotter.retrieve_tree(0)
# treePlotter.create_plot(myTree)
# print(classify(myTree,labels,[1,0]))    #输出no
# print(classify(myTree,labels,[1,1]))    #输出yes
# print(classify(myTree,labels,[0,1]))    #输出no
# print(classify(myTree,labels,[0]))      #输出no
# print(classify(myTree,labels,[1]))      #报错
# print(classify(myTree,labels,[1，2]))      #报错

'''
使用pickle模块存储决策树
'''
def store_tree(inputtree, filename):
    # fw = open(filename, 'w')
    #由于是二进制，因此直接w会报错write() argument must be str, not bytes，改成‘wb+’
    fw = open( filename , 'wb+' )
    pickle.dump(inputtree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename,'rb+')
    # fr = open(filename)由于文件模式不正确会报错：'gbk' codec can't decode byte 0x80 in position 0: illegal multibyte sequence
    # 改成fr = open(filename,'rb+')
    return pickle.load(fr)

# myDat,labels=create_dataset()
# myTree=treePlotter.retrieve_tree(0)
# store_tree(myTree,'classifierStorage.txt')
# print(grab_tree('classifierStorage.txt'))
#将分类器存储在硬盘上，不用每次对数据进行分类的时候再重新学习一遍
