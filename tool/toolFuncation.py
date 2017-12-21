# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import time

def getFunName(fun):
    '''
    获得函数的名称
    '''
    if '__name__' in dir(fun):
        return fun.__name__
    return '[unkonw]'

def dynamicWraps(func):
    '''
    decorator 动态规划 装饰器
    '''
    cache={}
    from functools import wraps
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args]=func(*args)
        return cache[args]
    return wrap

    
def setTimeOut(fun, t=0):
    '''
    same to setTimeOut in JavaScript
    '''
    from threading import Timer
    thread = Timer(t,fun)
    thread.start()
    return thread

def setInterval(fun, inter, maxTimes=None):
    '''
    same to setInterval in JavaScript
    '''
    maxTimes = [maxTimes]
    def interFun(): 
        fun()
        if maxTimes[0] is not None:
            maxTimes[0] -= 1
            if maxTimes[0] <= 0:
                return 
        setTimeOut(interFun, inter)
    interFun()


if __name__ == "__main__":
    
    def fun():
        for i in range(10):
            print i
            time.sleep(1)
    from threading import Timer
    thread = Timer(0,fun)
    thread.start()
    pass