# @description: 朴素贝叶斯分类器
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/3 10:38

from numpy import *
import re
import operator
import feedparser

# 词表到向量的转换函数
def load_data_set():
    '''
    加载数据
    :return: 词表向量、类别标签集合
    '''
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1代表侮辱性文字， 0代表正常言论
    class_vec = [0, 1, 0, 1, 0, 1]

    return posting_list, class_vec


# 创建一个包含在所有文档中出现的不重复的列表
def create_vocab_list(data_set):

    # 创建一个空集
    vocab_set = set([])

    for document in data_set:
        vocab_set = vocab_set | set(document)

    # 创建两个集合的并集
    return list(vocab_set)


# 输入参数是一个文档，输出的是文档向量        这是文档的词集模型
def set_of_word_vec(vocab_list, input_set):

    # 创建一个所含向量都为0的向量
    vec = [0] * len(vocab_list)

    for word in input_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] = 1
        else:
            print("The word:%s is not in my vocalulary!" % word)

    return vec

# listOPosts,listClasses=load_data_set()
# myVocabList=create_vocab_list(listOPosts)
# print(myVocabList)
# print(set_of_word_vec(myVocabList,listOPosts[0]))
# print(set_of_word_vec(myVocabList,listOPosts[3]))


# 朴素贝叶斯分类器的训练,未考虑计算多个概率乘积来获得文档属于某个类别的概率。
def trainNB0(train_matrix, train_category):
    '''
    :param train_matrix:输入包含全部文档词向量训练矩阵
    :param train_category: 一个文档的词向量列表
    :return:该文档各个词所属类别的条件概率，该文档属于哪个类别的概率
    '''
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)

    # 初始化概率
    p0_num = zeros(num_words)
    p1_num = zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0

    # 向量相加
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    # 对每个元素做除法
    p1_vec = p1_num / p1_denom
    p0_vec = p0_num / p0_denom
    return p0_vec, p1_vec, p_abusive

# listOPosts,listClasses=load_data_set()
# myVocabList=create_vocab_list(listOPosts)
# trainMat=[]     #初始化训练集
# for postinDoc in listOPosts:    #for循环使用词向量来填充训练集列表
#     trainMat.append(set_of_word_vec(myVocabList,postinDoc))
# p0V,p1V,pAb=trainNB0(trainMat,listClasses)
# print(myVocabList)
# print(p0V)
# print(p1V)
# print(pAb)

# 朴素贝叶斯分类器的训练，根据现实情况修改了上述分类器
def trainNB1(train_matrix, train_category):
    '''
    :param train_matrix:输入包含全部文档词向量训练矩阵
    :param train_category: 一个文档的词向量列表
    :return:该文档各个词所属类别的条件概率，该文档属于哪个类别的概率
    '''
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)

    # 初始化概率 后面取对数时
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    # 向量相加
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    # 对每个元素做除法并取对数
    p1_vec = log(p1_num / p1_denom)
    p0_vec = log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive

# listOPosts,listClasses=load_data_set()
# myVocabList=create_vocab_list(listOPosts)
# trainMat=[]     #初始化训练集
# for postinDoc in listOPosts:    #for循环使用词向量来填充训练集列表
#     trainMat.append(set_of_word_vec(myVocabList,postinDoc))
# p0V,p1V,pAb=trainNB1(trainMat,listClasses)
# print(myVocabList)
# print(p0V)
# print(p1V)
# print(pAb)

# 朴素贝叶斯的分类函数
def classify(vec_classify, p0_vec, p1_vec, p_class):
    '''
    :param vec_classify:输入需要分类的词向量
    :param p0_vec:文档中的词向量属于类别0的条件概率
    :param p1_vec:文档中的词向量属于类别1的条件概率
    :param p_class:文档属于类别1的概率
    :return:
    '''
    p1 = sum(vec_classify * p1_vec) + log(p_class)
    p0 = sum(vec_classify * p0_vec) + log(1.0 - p_class)
    if p1 > p0:     #使用贝叶斯判断准则
        return 1
    else:
        return 0


# 朴素贝叶斯的词袋模型
# 如果一个词在文档中出现的次数不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息
def bag_of_word_vec(vocab_list, inpiut_set):
    vec = [0] * len(vocab_list)
    for word in inpiut_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] += 1    #每次遇到一个单词时不是将对应的值置1，而是将对应值自增1

    return vec


# 朴素贝叶斯的测试
def test():
    list_of_posts, list_classes = load_data_set()
    my_vocal_list = create_vocab_list(list_of_posts)

    train_matrix = []

    for post_in_doc in list_of_posts:
        train_matrix.append(set_of_word_vec(my_vocal_list, post_in_doc))

    p0_vec, p1_vec, p_abusive = trainNB1(array(train_matrix), array(list_classes))

    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_word_vec(my_vocal_list, test_entry))
    print(test_entry, 'classified as:', classify(this_doc,p0_vec, p1_vec, p_abusive))

    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_word_vec(my_vocal_list, test_entry))
    print(test_entry, 'classified as: ', classify(this_doc, p0_vec, p1_vec, p_abusive))

# test()

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

    vocab_list = create_vocab_list(doc_list)
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
        train_matrix.append(bag_of_word_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
        p0_vec, p1_vec, p_spam = trainNB1(array(train_matrix), array(train_class))
        error_count = 0

        for doc_index in test_set:
            word_vector = bag_of_word_vec(vocab_list, doc_list[doc_index])
            if classify(array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
                error_count += 1
                print("classification error", doc_list[doc_index])

        print('the error rate is: ', float(error_count) / len(test_set))

# spam_text()
# 存在问题：某个邮件在处理的时候出现log(1.0 - p_class)中的真数为0的情况导致报错
# RuntimeWarning: divide by zero encountered in log
#   p0 = sum(vec_classify * p0_vec) + log(1.0 - p_class)

# RSS源分类器和高频去除函数
def calc_most_freq(vocab_list, full_text):
    '''
    遍历词汇表中的每个词并统计它在文本中出现的次数，从高到低对词典排序，最后返回排序最高的100个单词
    :param vocab_list:
    :param full_text:
    :return:
    '''
    freq_dict = {}
    # 计算出现频率
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)

    sorted_freq = sorted(freq_dict.items(),
                        key=operator.itemgetter(1), reverse=True)

    return sorted_freq[:30]


def local_words(feed0, feed1):
    doc_list = []
    class_list = []
    full_text = []

    min_len = min(len(feed1['entries']), len(feed0['entries']))

    # 每次访问一条RSS源
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)  # NY is class 1

        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)

    # 去掉那一些出现频率最高的词
    top_words = calc_most_freq(vocab_list, full_text)

    for pair_w in top_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])

    train_set = list(range(2*min_len))
    test_set = []

    for i in range(20):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])

    train_matrix = []
    train_class = []

    for docIndex in train_set:
        train_matrix.append(bag_of_word_vec(vocab_list, doc_list[docIndex]))
        train_class.append(class_list[docIndex])

    p0_vec, p1_vec, p_spam = trainNB1(array(train_matrix), array(train_class))
    error_count = 0

    for doc_index in test_set:
        word_vector = bag_of_word_vec(vocab_list, doc_list[doc_index])
        if classify(array(word_vector), p0_vec, p1_vec, p_spam) != class_list[doc_index]:
            error_count += 1

    print('the error rate is: ', float(error_count)/len(test_set))

    return vocab_list, p0_vec, p1_vec
# 注意作者提供的RSS源不可访问了，改用了两个可用源
# 所有来自于 nasa 的文章将会被分类为1，所有来自于yahoo sports的休斯顿火箭队新闻将会分类为0
# ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
# sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
# vocabList,pSF,pNY=local_words(ny,sf)



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

# ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
# sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
#
# get_top_words(ny,sf)
# 访问过于频繁会导致IP被封
