
from ylimg import *
from tool import *
import ylimg as tif
import logging
import argparse
import os
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np
import mxnet as mx

from collections import namedtuple
from time import time as T

def evaluation(gt,resoult):
    re = resoult>0.5
    TPl = (re==0)*(gt==0) #(re+gt) == 0
    FPl = (re==0)*(gt==1)
    FNl = (re==1)*(gt==0)
    

    TPs = (re==1)*(gt==1) 
    FPs = (re==1)*(gt==0)
    FNs = (re==0)*(gt==1)
#    show([TPs,FPs,FNs])
    TPl,FPl,FNl,TPs,FPs,FNs = [float(i.sum()) for i in [TPl,FPl,FNl,TPs,FPs,FNs]]
    LP = TPl/(TPl+FPl)
    LR = TPl/(TPl+FNl)
    OP = (TPl+TPs)/(TPl+FPl+TPs+FPs)
    OR = (TPl+TPs)/(TPl+FNl+TPs+FNs)
    return (LP*100,LR*100,OP*100,OR*100)
def evalImgs(model,evalNames,smooth=True):
    
    evalDic = {}
    for name in evalNames:
        predict(name,model,evalDic,smooth)
    evalNp = np.array(evalDic.values())
    eva = np.mean(evalNp,0)
    return eva
def predict(filename, evaluationDic=None,):
    img = tif.imread(filename)
    imgg = img
    img = img.astype(np.float32)
    img /= 255
    h, w, _ = img.shape
    hh, ww = h, w
    # h /= 2
    # w /= 2
    h = int(round(h / step) * step)
    w = int(round(w / step) * step)
    img = cv2.resize(img, (w, h))
    img = np.transpose(img, (2, 0, 1))
    label = np.zeros((h, w), dtype=np.int32)
    map = np.zeros((h, w), dtype=np.uint8)
    for x in range(0, w, step):
        for y in range(0, h, step):
            mod.forward(Batch(data=[mx.nd.array(np.expand_dims(
                img[:, y:y + step, x:x + step], 0))]), is_train=False)
            prob = mod.get_outputs()[0].asnumpy()
            label[y:y + step, x:x +
                  step] = np.argmax(np.squeeze(prob), axis=0)
    map[label == 0] = 0
    map[label == 1] = 255
#        map[label == 2, :] = (0, 255, 255)
#        map[label == 3, :] = (0, 255, 0)
#        map[label == 4, :] = (255, 255, 0)
#        map[label == 5, :] = (255, 0, 0)
    label = cv2.resize(label, (ww, hh), interpolation=cv2.INTER_NEAREST)!=0
    if evaluationDic is not None:
        label = label > 0.5
        gt = imread(filename.replace('.jpg','.png'))>0.5
        evaluationDic[name] = evaluation(gt,label)
        show(imgg,gt!=label)
        show(label,gt)
        print name,'LP=%.2f, LR=%.2f, OP=%.2f, OR=%.2f'%evaluationDic[name]
    return label
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--restore_step',
        type=int,
        default=1,
        help='params to restore'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=64*7,
        help='fixed step in the test phase'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default='./../sealand2/',
        help='test folder'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='unet',
        help='checkpoint prefix'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='out',
        help='dir to restore results'
    )
    args = parser.parse_args()
    step = args.step

    sym, arg_params, aux_params = mx.model.load_checkpoint(
        args.prefix, args.restore_step)
    # print(sym.list_outputs())
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.gpu())
    mod.bind(for_training=False, data_shapes=[
             ('data', (1, 3, step, step))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    Batch = namedtuple('Batch', ['data'])
    imgns = [i for i in os.listdir(args.test_path) if i[-4:]=='.jpg']
    
    dic = {}
    for img_name in imgns[:]:
        start = T()
        name = img_name[:-9]
        print('test {}'.format(img_name))
        filename = os.path.join(args.test_path, img_name)
        label = predict(filename,dic)
        imsave(os.path.join(args.out_dir, img_name[:-4]+'_class.png'),
                   label)
#        print('save to {}.tif, using {}s'.format(img_name[:-4]+'_class.tif', T() - start))
    