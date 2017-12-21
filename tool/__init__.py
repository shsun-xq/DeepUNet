# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os, sys, time

from toolLog import (stdout, log, tounicode, ignoreWarning,
                     LogException, LogLoopTime, SuperG, g, cf)
from toolLog import colorFormat, pblue, pred, pdanger, perr, pinfo
from toolLog import localTimeStr, gmtTimeStr

from toolDataStructureAndObject import (dicToObj, dicto, 
                                        listToBatch, FunAddMagicMethod)

from toolIo import (listdir, filename, openread, openwrite, replaceTabInPy, save_data, 
                    load_data, fileJoinPath)

from toolSystem import importAllFunCode, crun, frun, getArgvDic

from toolFuncation import getFunName, dynamicWraps, setInterval, setTimeOut

from toolTools import filterList, findints

from glob import glob
from collections import namedtuple
from functools import reduce
from os.path import join as pathjoin
from os.path import basename, isfile, isdir, dirname
from operator import add, sub, mul, div

if __name__ == "__main__":
    p = False
#    p = True
    if p:
        import toolTools as tt
        ttl = dir(tt)
        print(ttl)
        
        import toolFuncation as tf
        tfl = dir(tf)
        print(tf)
        
        import toolSystem as ts
        tsl = dir(ts)
        print(tsl)
        
        import toolIo as ti
        til = dir(ti)
        print(til)
        
        import toolDataStructureAndObject as tdo
        tdol = dir(tdo)
        print(tdol)
        
        import toolLog as tl
        tll = dir(tl)
        print(tll)
        
        import tool
        tooll = dir(tool)
        print(tooll)
    pass