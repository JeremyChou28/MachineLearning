# @description:使用决策树预测隐形眼镜类型
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/3 10:38
# 收集数据：提供的lenses文本文件
# 准备数据：解析tab键分隔的数据行
# 分析数据：快速检查数据，确保正确的解析数据内容，使用create_plot函数绘制决策树
# 训练算法：使用create_tree函数
# 测试算法：编写测试函数验证决策树可以正确的分类给定的数据实例
# 使用算法：存储决策树的数据结构，以便下次使用时无需重新构造决策树
import trees
import treePlotter

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=treePlotter.retrieve_tree(2)
# print(lensesTree)   #输出字典型决策树

# treePlotter.create_plot(lensesTree)   #绘制决策树
# print(lensesLabels)   #输出标签
print(trees.classify(lensesTree,lensesLabels,['young','hyper','yes','normal'])) #测试向量从下往上
trees.store_tree(lensesTree,'contactLensStorage.txt')   #存储决策树
print(trees.grab_tree('contactLensStorage.txt'))    #读取决策树