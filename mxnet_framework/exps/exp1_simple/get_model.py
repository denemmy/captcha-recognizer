import sys
from os.path import join, isdir, basename, splitext, dirname, abspath
import numpy as np
import mxnet as mx
from symbol_resnet import resnet

def get_loss(net, cfg):

    output_list = []
    for i in range(cfg.MAX_LABELS):
        fc_i = mx.symbol.FullyConnected(data=net, num_hidden=cfg.CLS_NUM, name='fc_{}'.format(i))
        top = mx.symbol.SoftmaxOutput(data=fc_i, name='softmax{}'.format(i), use_ignore=True)
        output_list.append(top)

    n_symbols_possible_cls = cfg.MAX_LABELS - cfg.MIN_LABELS + 1
    fc_n = mx.symbol.FullyConnected(data=net, num_hidden=n_symbols_possible_cls, name='fc_n')
    top = mx.symbol.SoftmaxOutput(data=fc_n, name='softmax{}'.format(cfg.MAX_LABELS))
    output_list.append(top)

    return mx.symbol.Group(output_list)

def get_model_pretrained(cfg):
    prefix, epoch = cfg.TRAIN.PRETRAINED, cfg.TRAIN.PRETRAINED_EPOCH

    # label = mx.sym.var('label')
    sym_model, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    all_layers = sym_model.get_internals()
    net = all_layers['flatten0_output']

    top = get_loss(net, cfg)

    return top, arg_params, aux_params

def get_model(cfg):
    # label = mx.sym.var('label')

    depth = 28
    k = 2

    per_unit = [(depth - 4) // 6]
    filter_list = [16, 16 * k, 32 * k, 64 * k]
    bottle_neck = False
    units = per_unit * 3

    net = resnet(units=units, num_stage=3, filter_list=filter_list, bottle_neck=bottle_neck)

    top = get_loss(net, cfg)

    return top

