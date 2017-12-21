# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from tool.toolDataStructureAndObject import FunAddMagicMethod
from tool.toolLog import colorFormat

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from skimage import io as sio
from skimage import data as da
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize 

# randomm((m, n), max) => m*n matrix
# randomm(n, max) => n*n matrix
randomm = lambda shape,maxx:(np.random.random(
shape if (isinstance(shape,tuple) or isinstance(shape,list)
)else (shape,shape))*maxx).astype(int)
r = randomm(4,4)

def normalizing(arr):
    a = arr.astype(float)
    minn = a.min()
    return (a-minn)/(a.max() - minn)
def uint8(img):
    '''将0～1的float或bool值的图片转换为uint8格式'''
    return ((img)*255.999).astype(np.uint8)

greyToRgb = lambda grey:grey.repeat(3).reshape(grey.shape+(3,)) 

npa = FunAddMagicMethod(np.array)


def mapp(f, matrix, need_i_j=False):
    '''
    for each item of a 2-D matrix
    return a new matrix consist of f:f(it) or f(it, i, j)
    性能差 尽量用y, x = np.mgrid[:10,:10]
    '''
    m, n = matrix.shape[:2]
    listt = [[None]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            it = matrix[i][j]
            listt[i][j] = f(it,i,j) if need_i_j else f(it)
    return np.array(listt)


from operator import add

def ndarrayToImgLists(arr):
    '''
    将所有ndarray转换为imgList
    '''
    arr = np.squeeze(arr)
    ndim = arr.ndim
    if arr.ndim==2 or (arr.ndim ==3 and arr.shape[-1] in [3,4]):
         return [arr]
    if arr.shape[-1] == 2: # 二分类情况下自动转换
        arr = arr.transpose(range(ndim)[:-3]+[ndim-1,ndim-3,ndim-2])
    imgdim = 3 if arr.shape[-1] in [3,4] else 2
    ls = list(arr)
    while ndim-1>imgdim:
        ls = reduce(add,map(list,ls),[])
        ndim -=1
    return ls
def listToImgLists(l, res=None):
    '''
    将 ndarray和list的混合结果树转换为 一维 img list
    '''
    if res is None:
        res = []
    for x in l:
        if isinstance(x,(list,tuple)):
            listToImgLists(x,res)
        if isinstance(x,dict):
            listToImgLists(x.values(),res)
        if isinstance(x,np.ndarray):
            res.extend(ndarrayToImgLists(x))
    return res
def showImgLists(imgs,**kv):
    n = len(imgs)
    if n == 4:
        showImgLists(imgs[:2],**kv)
        showImgLists(imgs[2:],**kv)
        return
    if n > 4:
        showImgLists(imgs[:3],**kv)
        showImgLists(imgs[3:],**kv)
        return 
    fig, axes = plt.subplots(ncols=n)
    count = 0
    axes = [axes] if n==1 else axes 
    for img in imgs:
        axes[count].imshow(img,**kv)
        count += 1
    plt.show()
def show(*imgs,**kv):
    '''
    do plt.imshow to a list of imgs or one img or img in dict or img in np.ndarray
    **kv: args for plt.imshow
    '''
    if 'cmap' not in kv:
        kv['cmap'] = 'gray'
    imgls = listToImgLists(imgs)
    showImgLists(imgls,**kv)
show = FunAddMagicMethod(show)


def showb(*arr,**__kv):
    '''
    use shotwell to show picture
    Parameters
    ----------
    arr : np.ndarray or path
    '''
    
    if len(arr)!=1:
        map(lambda ia:showb(ia[1],tag=ia[0]),enumerate(arr))
        return 
    arr = arr[0]
    if isinstance(arr,np.ndarray):
        path = '/tmp/tmp-%s.png'%len(glob.glob('/tmp/tmp-*.png'))
        imsave(path,arr)
        arr = path
    cmd = 'shotwell "%s" &'%arr
    os.system(cmd)
showb = FunAddMagicMethod(showb)

def loga(array):
    '''
    Analysis np.array with a graph. include shape, max, min, distribute
    '''
    if isinstance(array,list):
        array = np.array(array)
    if isinstance(array,str) or isinstance(array,unicode):
        print 'info and histogram of',array
        l=[]
        eval('l.append('+array+')')
        array = l[0]
    print 'shape:%s ,type:%s ,max: %s, min: %s'%(str(array.shape),array.dtype.type, str(array.max()),str(array.min()))
    
    unique = np.unique(array)
    if len(unique)<10:
        dic=dict([(i*1,0) for i in unique])
        for i in array.ravel():
            dic[i] += 1
        listt = dic.items()
        listt.sort(key=lambda x:x[0])
        data,x=[v for k,v in listt],np.array([k for k,v in listt]).astype(float)
        if len(x) == 1:
            print 'All value is',x[0]
            return
        width = (x[0]-x[1])*0.7
        x -=  (x[0]-x[1])*0.35
    else:
        data, x = np.histogram(array.ravel(),8)
        x=x[1:]
        width = (x[0]-x[1])
    plt.plot(x, data, color = 'orange')
    plt.bar(x, data,width = width, alpha = 0.5, color = 'b')
    plt.show()
    return 

loga = FunAddMagicMethod(loga)


selfFuns = {
 list:lambda x:colorFormat.b%('%d*[]'%len(x)),
 tuple:lambda x:colorFormat.b%('%d*()'%len(x)),
 dict:lambda x:colorFormat.b%('keys:{%s}'%str(list(x.keys()))[1:-1]),
 np.ndarray:lambda x:colorFormat.r%('%s%s'%
                                    (str(x.shape).replace('L,','').replace('L',''),x.dtype)),
 }
def selfToStr(se):
    for typ in selfFuns:
        if isinstance(se,typ):
            strr = selfFuns[typ](se)
            break
    else:
        strr = colorFormat.r%str(se)
    return strr
def logl(listt):
    '''
    简单查看list, tuple, dict, numpy组成的树的每一层结构
    可迭代部分用蓝色 叶子用红色打印
    '''
    l = listt
    strss = [['0.  '+selfToStr(l)]]
    def logll(now,strss):
        new = []
        strs = [str(len(strss))+'.']
        for l in now:
            if isinstance(l,dict):
                new.extend(l.values())
                strs += ['{%s}'%', '.join(list(map(selfToStr,l.values())))]
            if isinstance(l,(list)):
                new.extend(l)
                strs += ['[%s]'%', '.join(list(map(selfToStr,l)))]
            if isinstance(l,(tuple)):
                new.extend(l)
                strs += ['(%s)'%', '.join(list(map(selfToStr,l)))]
        if len(strs)>1:
             strss.append(strs)
        if len(new):
            logll(new,strss)
    logll([l],strss)
    strss = '\n'.join(['  '.join(xs) for xs in strss])
    print(strss)

    
def labelToColor(label,colors):
    '''
    将颜色映射到label上
    '''
    rgb = np.zeros(label.shape+(3,))
    for c in np.unique(label):
        rgb[label==c] = colors[c]
    return rgb

def standImg(img):
    '''
    任何输入img都转换为 shape is (m, n, 3) dtype == Float
    '''
    from ylnp import isNumpyType
    if img.ndim == 2:
        img = greyToRgb(img)
    if img.dtype == np.uint8:
        img = img/255.
    if isNumpyType(img, bool):
        img = img*1.0
    if isNumpyType(img, float):
        return img
    
def generateBigImgForPaper(imgMa,lengh=1980,border=20,saveName='bigImgForPaper.png'):
    '''
    生成科研写作用的样本对比图
    imgMa: 图片行程的二维列表
    lengh: 大图片的宽度, 长度根据imgMa矩阵的高自动算出
    border: 图片与图片间的间隔
    '''
    big = None
    for rr in imgMa:
        rr = map(standImg,rr)
        nn = len(rr)
        a = int((lengh-nn*border)/nn)
        m,n = rr[0].shape[:2]
        b = int(a*m/n)
        row = None
        rr = [resize(r,(b,a)) for r in rr]
        for r in rr:

            if row is None:
                row = r
            else:
                row = np.append(row,np.ones((b,border,3)),1)
                row = np.append(row,r,1)
        if big is None:
            big = row
        else:
            big = np.append(big,np.ones((border,big.shape[1],3)),0)
            big = np.append(big,row,0)

    show(big)
    if saveName:
        imsave(saveName,big)
    return big
if __name__ == '__main__':

    pass
