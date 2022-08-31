# @description:使用朴素贝叶斯分类器从个人广告中获取区域倾向
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/3 17:39
# 收集数据：从RSS源收集内容，需要对RSS源构建一个接口
# 准备数据：将文本文件解析成词条向量
# 分析数据：检查词条确保解析的正确性
# 训练算法：使用建立的trainNB1()函数
# 测试算法：观察错误率，确保分类器可用。可以修改切分程序来降低错误率，提高分类结果
# 使用算法：构建一个完整程序，封装所有内容。给定两个RSS源，该程序会显示常用的公共词。
import bayes
import operator
from numpy import *

# RSS源分类器和高频去除函数
def calc_most_freq(vocab_list, full_text):
    freq_dict = {}

    for token in vocab_list:
        freq_dict[token] = full_text.count(token)

    sorted_freq = sorted(freq_dict.tems(),
                        key=operator.itemgetter(1), reverse=True)

    return sorted_freq[:30]


def local_words(feed0, feed1):
    doc_list = []
    class_list = []
    full_text = []

    min_len = min(len(feed1['entries']), len(feed0['entries']))

    # 每次访问一条RSS源
    for i in range(min_len):
        word_list = bayes.text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)  # NY is class 1

        word_list = bayes.text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = bayes.create_vocab_list(doc_list)

    # 去掉那一些出现频率最高的词
    top_words = calc_most_freq(vocab_list, full_text)

    for pair_w in top_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])

    train_set = range(2*min_len)
    test_set = []  #

    for i in range(20):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])

    train_matrix = []
    train_class = []

    for docIndex in train_set:
        train_matrix.append(bayes.bag_of_word_vec(vocab_list, doc_list[docIndex]))
        train_class.append(class_list[docIndex])

    p0_vec, p1_vec, p_spam = bayes.trainNB1(array(train_matrix), array(train_class))
    error_count = 0

    for doc_index in test_set:
        word_vector = bayes.bag_of_word_vec(vocab_list, doc_list[doc_index])
        if bayes.classify(array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            error_count += 1

    print('the error rate is: ', float(error_count)/len(test_set))

    return vocab_list, p0_vec, p1_vec


# 最具特征性的词汇显示函数
def get_top_words(ny, sf):
    vocab_list, p0_vec, p1_vec = local_words(ny, sf)
    top_ny = []
    top_sf = []

    for i in range(len(p0_vec)):
        if p0_vec[i] > -6.0:
            top_sf.append((vocab_list[i], p0_vec[i]))
        if p1_vec[i] > -6.0:
            top_ny.append((vocab_list[i], p1_vec[i]))

    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    print("sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**sf**")

    for item in sorted_sf:
        print(item[0])

    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print("ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**ny**")

    for item in sorted_ny:
        print(item[0])
