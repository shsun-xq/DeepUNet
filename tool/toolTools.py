# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import re, random

def filterList(key, strs):
    '''
    对一个str列表 找出其中存在字符串 key的所有元素
    '''
    return list(filter((lambda strr: key in strr),strs))

def findints(strr):
    '''
    返回字符串或字符串列表中的所有整数
    '''
    if isinstance(strr,(list,tuple)):
        return list(map(findint, strr))
    return list(map(int,re.findall(r"\d+\d*",strr)))

def randint(maxx=100):
    return random.randint(0, maxx)

def randfloat():
    return random.random()

def randchoice(seq, num=None):
    '''
    随机选择一个列表内的一个或num个元素
    '''
    if num is None:
        return random.choice(seq)
    return random.sample(seq, num)

if __name__ == "__main__":
     
    string=["A001.45，b5，6.45，8.82",'sd4 dfg77']
    print findint(string)
    pass