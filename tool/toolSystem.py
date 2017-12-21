# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys

def importAllFunCode(mod=None):
    '''
    mod 为包名(type(mod)=='str')或包本身(type(mod)==module)
    自动生成导入所有模块语句 并过滤掉__name__等
    '''
    if mod is None:
        mod = 'yllab'
    if isinstance(mod,(str,unicode)):
        exec ('import %s as mod'%mod)

    names = [name for name in dir(mod) if not ((len(name)>2 and name[:2]=='__') or 
                                  name in ['unicode_literals',])]
    n = 5
    lines = []
    while len(names) > n:
        l,names = names[:n],names[n:]
        lines += [', '.join(l)]
    lines += [', '.join(names)]
    lines = ',\n          '.join(lines)
    
    strr = (("from %s import *\nfrom %s import (%s)"%(mod.__name__,mod.__name__,lines)))
    print strr

def crun(pycode):
    '''测试代码pycode的性能'''
    from cProfile import run
    return run(pycode,sort='time')
def frun(pyFileName=None):
    '''在spyder中 测试pyFileName的性能'''
    if pyFileName:
        if '.py' not in pyFileName:
            pyFileName += '.py'
        crun("runfile('%s',wdir='.')"%pyFileName)
    else:
        crun("runfile(__file__,wdir='.')")


def strIsInt(s):
    '''判断字符串是不是整数型'''
    s = s.replace(' ','')
    return s.isdigit() or (s[0]==('-') and s[1:].isdigit())

def strIsFloat(s):
    '''判断字符串是不是浮点'''
    s = s.replace(' ','')
    return s.count('.')==1 and strIsInt(s.replace('.',''))
def strToNum(s):
    ''' 若字符串是float or int 是则返回 数字 否则返回本身'''
    if strIsInt(s):
        return int(s)
    if strIsFloat(s):
        return float(s)
    return s

def getArgvDic(argvTest=None):
    '''
    将`python main.py xxx --k v`形式的命令行参数转换为list, dict
    若v是数字 将自动转换为 int or float
    '''
    from toolLog import  pred
    argv = sys.argv
    if argvTest:
        argv = argvTest
    l = argv = map(strToNum,argv[1:])
    code = map(lambda x:(isinstance(x,(str,unicode)) 
        and len(x) >2 and x[:2]=='--'),argv)
    dic = {}
    if True in code:
        l = argv[:code.index(True)]
        n = len(code)
        for i,s in enumerate(code):
            x = argv[i]
            if int(s):
                k = x.replace('--','')
                if (i<=n-2 and code[i+1]) or i==n-1: # 不带参数
                    dic[k] = True
                else:  # 带参数
                    dic[k] = argv[i+1]
    if len(dic) or len(l):
        pred('command-line arguments are:\n  %s and %s'%(l,dic))
    return l,dic
    
if __name__ == "__main__":

    pass