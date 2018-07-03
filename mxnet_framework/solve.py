import sys
from os.path import join, isdir, basename, splitext, isfile
import numpy as np
import mxnet as mx
from os import mkdir
import argparse
from config import cfg, cfg_from_file
from iterators import ClsIter, MxIterWrapper
from callbacks import SpeedometerCustom, BestModelCheckpoint
from module import CustomModule
from mxnet import metric as mxmetric
from datasets import CollectionDataset
from image_sample import SampleWithCache
from metrics import CaptchaAccuracy

import logging
logging.getLogger().setLevel(logging.INFO)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--gpus', dest='gpu_ids',
                        help='GPU device id to use 0,1,2', type=str)
    parser.add_argument('--exp-name', dest='exp_name', required=True, type=str)
    parser.add_argument('--resume_model', dest='resume_model', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def solve():

    # init seed
    np.random.seed(cfg.RNG_SEED)
    mx.random.seed(cfg.RNG_SEED)

    # parse cli arguments
    args = parse_args()

    print('called with args:')
    print(args)

    exp_name = args.exp_name
    exp_dir = join('exps', exp_name)

    snapshots_dir = join(exp_dir, 'snapshots')
    if not isdir(snapshots_dir):
        mkdir(snapshots_dir)
    prefix = '{}/'.format(snapshots_dir)

    # load config
    cfg_file = join(exp_dir, 'config.yml')
    cfg_from_file(cfg_file)

    cache_max_size = cfg.CACHE_MAX_SIZE
    SampleWithCache.set_cache_size(cache_max_size)

    if args.gpu_ids:
        cfg.GPU_IDS = [int(gpu) for gpu in args.gpu_ids.split(',')]

    if len(cfg.GPU_IDS) == 0:
        # use all available GPU
        devices = [mx.gpu(i) for i in range(cfg.GPU_NUM)]
    else:
        devices = [mx.gpu(i) for i in cfg.GPU_IDS]

    # init iterators
    train_dataset = CollectionDataset(cfg.TRAIN.DATASETS, cfg)
    train_n_samples = len(train_dataset)
    if train_n_samples <= 0:
        print('number of training samples should be > 0')
        exit(-1)

    train_iter = MxIterWrapper(train_dataset, cfg, num_workers=cfg.TRAIN.LOADER_WORKERS)
    batch_size_train = train_iter.batch_size
    iters_per_epoch = int(train_n_samples / cfg.TRAIN.BATCH_SIZE)
    iters_per_epoch += 1 if train_n_samples % cfg.TRAIN.BATCH_SIZE > 0 else 0

    if len(cfg.VALIDATION.DATASETS) > 0:
        val_dataset = CollectionDataset(cfg.VALIDATION.DATASETS, cfg, test_cfg=cfg.VALIDATION)
        val_iter = MxIterWrapper(val_dataset, cfg, test_cfg=cfg.VALIDATION, num_workers=cfg.VALIDATION.LOADER_WORKERS) if len(val_dataset) else None
        val_n_samples = len(val_dataset)
    else:
        val_n_samples = 0

    print('total train samples: {}'.format(train_n_samples))
    print('total validation samples: {}'.format(val_n_samples))
    print('batch size: {}'.format(cfg.TRAIN.BATCH_SIZE))
    print('epoch size: {}'.format(iters_per_epoch))

    resume_model = False
    if args.resume_model:
        sys.path.insert(0, exp_dir)
        from get_optimizer_params import get_optimizer_params
        epoch = int(args.resume_model)
        sym_model, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        initw, optimizer, optimizer_params = get_optimizer_params(train_n_samples, cfg, begin_epoch=epoch)
        resume_model = True
    else:
        sys.path.insert(0, exp_dir)
        from get_optimizer_params import get_optimizer_params
        from get_model import get_model, get_model_pretrained
        if cfg.TRAIN.PRETRAINED:
            sym_model, arg_params, aux_params = get_model_pretrained(cfg)
        else:
            sym_model = get_model(cfg)
            arg_params = None
            aux_params = None
        initw, optimizer, optimizer_params = get_optimizer_params(train_n_samples, cfg)

    if cfg.PLOT_GRAPH:
        graph_shapes = {}
        graph_shapes['data'] = train_iter.provide_data[0][1]
        for i in range(cfg.MAX_LABELS):
            graph_shapes['softmax{}_label'.format(i)] = train_iter.provide_label[i][1]
        graph = mx.viz.plot_network(symbol=sym_model, shape=graph_shapes)
        graph.format = 'png'
        graph.render('{}/graph'.format(exp_dir))

    display = cfg.TRAIN.DISPLAY_ITERS
    snapshot_period = cfg.TRAIN.SNAPSHOT_PERIOD
    epochs_to_train = cfg.TRAIN.EPOCHS

    checkpoint_callback = mx.callback.do_checkpoint(prefix, snapshot_period)
    epoch_end_callbacks = [checkpoint_callback]

    # metrics
    acc_metric = CaptchaAccuracy(cfg)
    eval_metric = acc_metric

    val_acc_metric = CaptchaAccuracy(cfg)
    validation_metric = val_acc_metric

    metric_checkpoint_name = 'accuracy'

    best_model_prefix = '{}/model-best'.format(snapshots_dir)
    best_model_checkpoint_callback = BestModelCheckpoint(best_model_prefix, metric_checkpoint_name)
    best_model_checkpoint_callbacks = [best_model_checkpoint_callback]

    eval_interval = cfg.VALIDATION.INTERVAL if cfg.VALIDATION.REPORT_INTERMEDIATE else None
    label_names = ['softmax{}_label'.format(i) for i in range(cfg.MAX_LABELS+1)]
    model = CustomModule(symbol=sym_model, label_names=label_names,
                         context=devices)
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    if resume_model:
        model.init_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)

        model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  eval_metric=eval_metric,  # report accuracy during training
                  validation_metric=validation_metric,
                  batch_end_callback=SpeedometerCustom(batch_size_train, display), # output progress for each N data batches
                  epoch_end_callback=epoch_end_callbacks,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  begin_epoch=epoch,
                  eval_interval=eval_interval,
                  best_model_callbacks=best_model_checkpoint_callbacks,
                  num_epoch=epochs_to_train)
    else:
        model.init_params(initw, arg_params=arg_params, aux_params=aux_params, allow_missing=True)
        model.init_optimizer(optimizer=optimizer, optimizer_params=optimizer_params)

        model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  eval_metric=eval_metric,  # report metric values during training
                  validation_metric=validation_metric,
                  batch_end_callback=SpeedometerCustom(batch_size_train, display), # output progress for each 100 data batches
                  epoch_end_callback=epoch_end_callbacks,
                  eval_interval=eval_interval,
                  best_model_callbacks=best_model_checkpoint_callbacks,
                  num_epoch=epochs_to_train)

    model.save_checkpoint(prefix, epochs_to_train)

if __name__ == '__main__':
    solve()

