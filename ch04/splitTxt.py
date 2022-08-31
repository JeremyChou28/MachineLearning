# @description:切分文本
# @Author: 周健平
# @company: 山东大学
# @Time: 2020/10/3 17:08
import re

mySent='This book is the best book on python or M.L. I have ever laid eyes upon.'
# print(mySent.split())   #标点符号也被当成了词的一部分
# print(mySent)

regEx=re.compile("\\w*")
listOfTokens=regEx.split(mySent)
print(listOfTokens)