import sys
from os.path import join, isdir, basename, splitext, dirname, abspath
import numpy as np
import mxnet as mx
from symbol_resnet import resnet

def lsoftmax_loss(bottom, label, cls_num, name):

    MARGIN = 4
    BETA = 100.
    BETA_MIN = 4.
    SCALE = 0.9993

    fc1 = mx.symbol.LSoftmax(data=bottom,
                             num_hidden=cls_num,
                             label=label,
                             margin=MARGIN,
                             beta=BETA,
                             beta_min=BETA_MIN,
                             scale=SCALE,
                             name='b_{}'.format(name))

    top = mx.symbol.SoftmaxOutput(data=fc1, label=label, name=name)
    return top

def softmax_loss(bottom, label, cls_num, name):
    fc1 = mx.symbol.FullyConnected(data=bottom, num_hidden=cls_num, name='fc_{}'.format(name), no_bias=True)
    top = mx.symbol.SoftmaxOutput(data=fc1, label=label, name=name)
    return top

def get_output(net, labels, cfg):

    output_list = []
    for i in range(cfg.MAX_LABELS):
        top = softmax_loss(net, labels[i], cfg.CLS_NUM, 'softmax{}'.format(i))
        output_list.append(top)

    # n_symbols_possible_cls = cfg.MAX_LABELS - cfg.MIN_LABELS + 1
    # top = softmax_loss(net, labels[-1], n_symbols_possible_cls, 'softmax{}'.format(cfg.MAX_LABELS))
    # output_list.append(top)

    return mx.symbol.Group(output_list)

def get_model_pretrained(cfg):
    prefix, epoch = cfg.TRAIN.PRETRAINED, cfg.TRAIN.PRETRAINED_EPOCH

    labels = []

    # for i in range(cfg.MAX_LABELS + 1):
    #     label = mx.sym.var('softmax{}_label'.format(i))
    #     labels.append(label)

    for i in range(cfg.MAX_LABELS):
        label = mx.sym.var('softmax{}_label'.format(i))
        labels.append(label)
    sym_model, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    all_layers = sym_model.get_internals()
    net = all_layers['flatten0_output']

    top = get_output(net, labels, cfg)

    return top, arg_params, aux_params

def get_model(cfg):
    labels = []

    # for i in range(cfg.MAX_LABELS + 1):
    #     label = mx.sym.var('softmax{}_label'.format(i))
    #     labels.append(label)

    for i in range(cfg.MAX_LABELS):
        label = mx.sym.var('softmax{}_label'.format(i))
        labels.append(label)

    depth = 28
    k = 2

    per_unit = [(depth - 4) // 6]
    filter_list = [16, 16 * k, 32 * k, 64 * k]
    bottle_neck = False
    units = per_unit * 3

    net = resnet(units=units, num_stage=3, filter_list=filter_list, bottle_neck=bottle_neck)

    top = get_output(net, labels, cfg)

    return top

