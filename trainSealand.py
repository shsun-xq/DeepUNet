# coding: utf-8
from ylimg import *
from tool import *
import random
import logging
import argparse
import os
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np
import mxnet as mx
import scipy.io as sio
import matplotlib.pyplot as plt
import tifffile as tif


class SimpleBatch(object):

    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad


class DataGen():

    def __init__(self, image, label):
        self._images = image
        self._labels = label
        self._step = 64 * 14
        self._step = 64 * 7
        self._flip = True

    def get_data_label(self, batch):
        step = self._step
        images = []
        labels1 = []
        labels2 = []
        labels3 = []
        labels4 = []
        for _ in range(batch):
            idx = random.randint(0, len(self._images) - 1)
            self._image = self._images[idx]
            self._label = self._labels[idx]
            self._h, self._w, _ = self._image.shape
            y = random.randint(0, self._h - step - 1)
            x = random.randint(0, self._w - step - 1)
#            log(x,y,step,self._h)
            image = self._image[y:y + step, x:x + step, :]
            label = self._label[y:y + step, x:x + step]
            image = cv2.resize(image, (self._step, self._step))
#            g.x=label,g
            label = cv2.resize(label, (self._step, self._step),
                               interpolation=cv2.INTER_NEAREST)
            # filp
            if self._flip:
                if random.random() > 0.5:
                    image = np.fliplr(image)
                    label = np.fliplr(label)
                if random.random() > 0.5:
                    image = np.flipud(image)
                    label = np.flipud(label)

            # preprocess
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            # adjust brightness
            hsv[:, :, 2] += random.randint(-15, 15)
            # adjust saturation
            hsv[:, :, 1] += random.randint(-10, 10)
            # adjust hue
            hsv[:, :, 0] += random.randint(-5, 5)
            hsv = np.clip(hsv, 0, 255)
            hsv = hsv.astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            image = image.astype(np.float32)
            image /= 255.0

            # plt.subplots(1, 2)
            # plt.subplot(1,2,1)
            # plt.imshow(image)
            # plt.subplot(1,2,2)
            # plt.imshow(label)
            # plt.show()

            image = np.transpose(image, (2, 0, 1))

            images.append(image)
            labels1.append(label)
            labels2.append(cv2.resize(label, (self._step // 2, self._step // 2),
                                      interpolation=cv2.INTER_NEAREST))
            labels3.append(cv2.resize(label, (self._step // 4, self._step // 4),
                                      interpolation=cv2.INTER_NEAREST))
            labels4.append(cv2.resize(label, (self._step // 8, self._step // 8),
                                      interpolation=cv2.INTER_NEAREST))
        return np.stack(images), np.stack(labels1), np.stack(labels2), np.stack(labels3), np.stack(labels4)


class SimpleIter:

    def __init__(self, data_names, data_shapes,
                 label_names, label_shapes, gen, batchsize, num_batches=10000):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = gen
        self.cur_batch = 0
        self._batchsize = batchsize

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            img, label1, label2, label3, label4 = self.data_gen.get_data_label(
                self._batchsize)
            data = [mx.nd.array(img)]
            assert len(data) > 0, "Empty batch data."
            label = []
            label.append(mx.nd.array(label1))
            label.append(mx.nd.array(label2))
            label.append(mx.nd.array(label3))
            label.append(mx.nd.array(label4))
            assert len(label) > 0, "Empty batch label."
            return SimpleBatch(data, label)
        else:
            raise StopIteration


def conv(data, kernel=(3, 3), stride=(1, 1), pad=(0, 0), num_filter=None, name=None):
    return mx.sym.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter, name='conv_{}'.format(name))


def bn_relu(data, name):
    return mx.sym.Activation(data=mx.sym.BatchNorm(data=data, momentum=0.99, name='bn_{}'.format(name)), act_type='relu', name='relu_{}'.format(name))


def conv_bn_relu(data, kernel=(3, 3), stride=(1, 1), pad=(0, 0), num_filter=None, name=None):
    return bn_relu(conv(data, kernel, stride, pad, num_filter, 'conv_{}'.format(name)), 'relu_{}'.format(name))


def down_block(data, f, name):
    x = mx.sym.Pooling(data=data, kernel=(2,2), stride=(2,2), pool_type='max')
    # temp = conv_bn_relu(data, (3, 3), (2, 2), (1, 1),
    #                     f, 'layer1_{}'.format(name))
    temp = conv_bn_relu(x, (3, 3), (1, 1), (1, 1),
                        2*f, 'layer2_{}'.format(name))
    bn = mx.sym.BatchNorm(data=conv(temp, (3, 3), (1, 1), (1, 1), f, 'layer3_{}'.format(
        name)), momentum=0.99, name='layer3_bn_{}'.format(name))
    bn += x
    act = mx.sym.Activation(data=bn, act_type='relu',
                            name='layer3_relu_{}'.format(name))
    return bn, act


def up_block(act, bn, f, name):
    x = mx.sym.UpSampling(
        act, scale=2, sample_type='nearest', name='upsample_{}'.format(name))
    # temp = mx.sym.Deconvolution(data=act, kernel=(3, 3), stride=(2, 2), pad=(
    #    1, 1), adj=(1, 1), num_filter=32, name='layer1_dconv_{}'.format(name))
    temp = mx.sym.concat(bn, x, dim=1)
    temp = conv_bn_relu(temp, (3, 3), (1, 1), (1, 1),
                        2*f, 'layer2_{}'.format(name))
    bn = mx.sym.BatchNorm(data=conv(temp, (3, 3), (1, 1), (1, 1), f, 'layer3_{}'.format(
        name)), momentum=0.99, name='layer3_bn_{}'.format(name))
    bn += x
    act = mx.sym.Activation(data=bn, act_type='relu',
                            name='layer3_relu_{}'.format(name))
    return act


def get_net():
    data = mx.sym.Variable('data')
    x = conv_bn_relu(data, (3, 3), (1, 1), (1, 1), 32, 'conv0_1')
    net = conv_bn_relu(x, (3, 3), (1, 1), (1, 1), 64, 'conv0_2')
    bn1 = mx.sym.BatchNorm(data=conv(
        net, (3, 3), (1, 1), (1, 1), 32, 'conv0_3'), momentum=0.99, name='conv0_3_bn')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name='conv0_3_relu')

    bn2, act2 = down_block(act1, 32, 'down1')
    bn3, act3 = down_block(act2, 32, 'down2')
    bn4, act4 = down_block(act3, 32, 'down3')
    bn5, act5 = down_block(act4, 32, 'down4')
    bn6, act6 = down_block(act5, 32, 'down5')
    bn7, act7 = down_block(act6, 32, 'down6')

    temp = up_block(act7, bn6, 32, 'up6')
    temp = up_block(temp, bn5, 32, 'up5')
    temp = up_block(temp, bn4, 32, 'up4')
    score4 = conv(temp, (1, 1), (1, 1), (0, 0), 2, 'score4')
    net4 = mx.sym.SoftmaxOutput(score4, multi_output=True, name='softmax4')

    temp = up_block(temp, bn3, 32, 'up3')
    score3 = conv(temp, (1, 1), (1, 1), (0, 0), 2, 'score3')
    net3 = mx.sym.SoftmaxOutput(score3, multi_output=True, name='softmax3')

    temp = up_block(temp, bn2, 32, 'up2')
    score2 = conv(temp, (1, 1), (1, 1), (0, 0), 2, 'score2')
    net2 = mx.sym.SoftmaxOutput(score2, multi_output=True, name='softmax2')

    temp = up_block(temp, bn1, 32, 'up1')
    score1 = conv(temp, (1, 1), (1, 1), (0, 0), 2, 'score1')
    net1 = mx.sym.SoftmaxOutput(score1, multi_output=True, name='softmax1')

    return mx.sym.Group([net1, net2, net3, net4])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batchsize',
        type=int,
        default=12,
        help='number of training image per batch'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10000,
        help='number of training image per batch'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=50,
        help='number of training image per batch'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='number of gpu to use'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='plot the network structure'
    )
    parser.add_argument(
        '--resume',
        type=int,
        default=0,
        help='which epoch to resume'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='unet',
        help='prefix of the model name'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2,
        help='learning rate'
    )

    args = parser.parse_args()

    net = get_net()

    if args.resume:
        print('resume training from epoch {}'.format(args.resume))
        _, arg_params, aux_params = mx.model.load_checkpoint(
            args.prefix, args.resume)
    else:
        arg_params = None
        aux_params = None

    if args.plot:
        mx.viz.plot_network(net, save_format='pdf', shape={
            'data': (1, 3, 640, 640),
            'softmax1_label': (1, 640, 640),
            'softmax2_label': (1, 320, 320),
            'softmax3_label': (1, 160, 160),
            'softmax4_label': (1, 80, 80), }).render(args.prefix)
        exit(0)

#%%
    imgNames = glob('../sealand/*.jpg')  
    labels = [imread(n.replace('.jpg','.png'))>0 for n in imgNames]
    labels = [l.astype(np.uint8) for l in labels]
    imgs = map(imread,imgNames)
    img,label = imgs[0], labels[0]
    images = imgs
#%%
    dg = DataGen(images, labels)

    b = args.batchsize

    mod = mx.mod.Module(
        symbol=net,
        context=[mx.gpu(k) for k in range(args.gpu)],
        data_names=('data',),
        label_names=('softmax1_label', 'softmax2_label',
                     'softmax3_label', 'softmax4_label',)
    )

    data = SimpleIter(('data',),
                      [(b, 3, dg._step, dg._step)],
                      ('softmax1_label', 'softmax2_label',
                       'softmax3_label', 'softmax4_label',),
                      [(b, dg._step, dg._step), (b, dg._step // 2, dg._step // 2),
                       (b, dg._step // 4, dg._step // 4), (b, dg._step // 8, dg._step // 8)],
                      dg,
                      b,
                      num_batches=args.num_batches)

    total_steps = args.num_batches * args.epoch
    lr_sch = mx.lr_scheduler.MultiFactorScheduler(
        step=[total_steps // 2, total_steps // 4 * 3], factor=0.1)

    mod.fit(
        data,
        begin_epoch=args.resume,
        arg_params=arg_params,
        aux_params=aux_params,
        batch_end_callback=mx.callback.Speedometer(b),
        epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
        optimizer='sgd',
        optimizer_params=(('learning_rate', args.lr), ('momentum', 0.9),
                          ('lr_scheduler', lr_sch), ('wd', 0.0005)),
        num_epoch=args.epoch)
