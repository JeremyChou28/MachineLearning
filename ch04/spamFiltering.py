# @description:利用朴素贝叶斯对电子邮件进行过滤
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/3 17:03
# 收集数据：提供文本文件
# 准备数据：将文本文件解析成词条向量
# 分析数据：检查词条确保解析的正确性
# 训练算法：使用之前建立的trainNB1()函数
# 测试算法：使用classify()，并且构建一个新的测试函数来计算文档集的错误率
# 使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上
import re
import bayes
from numpy import *
#使用朴素贝叶斯对电子邮件进行分类

# 切割分类文本
def text_parse(big_string):
    list_of_tokens = re.split('\W+', big_string)

    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


# 垃圾邮件检测
def spam_text():
    doc_list = []
    class_list = []
    full_text = []

    # 导入并且解析文本
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i,encoding='ISO 8859-1').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)

        word_list = text_parse(open('email/ham/%d.txt' % i,encoding='ISO 8859-1').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)

    vocab_list = bayes.create_vocab_list(doc_list)
    # train_set = range(50)   #报错：'range' object doesn't support item deletion
    #应改为train_set = list(range(50))
    train_set = list( range( 50 ) )
    test_set = []

    # 随机构建训练集
    for i in range(10):
        rand_index = int(random.uniform(1, len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])

    train_matrix = []
    train_class = []

    # 对测试集合进行分类
    for doc_index in train_set:
        train_matrix.append(bayes.bag_of_word_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
        p0_vec, p1_vec, p_spam = bayes.trainNB1(array(train_matrix), array(train_class))
        error_count = 0

        for doc_index in test_set:
            word_vector = bayes.bag_of_word_vec(vocab_list, doc_list[doc_index])
            if bayes.classify(array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
                error_count += 1
                print("classification error", doc_list[doc_index])

        print('the error rate is: ', float(error_count) / len(test_set))

spam_text()
# 存在问题：某个邮件在处理的时候出现log(1.0 - p_class)中的真数为0的情况导致报错
# RuntimeWarning: divide by zero encountered in log
#   p0 = sum(vec_classify * p0_vec) + log(1.0 - p_class)