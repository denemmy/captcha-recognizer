import sys
import numpy as np
import mxnet as mx
from os import mkdir, makedirs
from os.path import join, isdir, basename, dirname, splitext
import argparse
import cv2
from config import cfg, cfg_from_file
from iterators import MxIterWrapper
import mxnet.metric as mxmetric
from tqdm import tqdm
from transform import convert_color_space
from metrics import CaptchaAccuracy
from datasets import CollectionDataset
from utils import get_symbols_from_codes

def visualize_result(images, imnames, gt_labels, pr_labels, output_dir, cfg, only_error=True):

    n_samples = len(images)
    assert(n_samples == len(imnames))
    assert(n_samples == gt_labels.shape[0])
    assert(n_samples == pr_labels.shape[0])

    for si in range(n_samples):

        vis_img = images[si]
        imname_orig = imnames[si]
        gt_label = gt_labels[si,:]
        pr_label = pr_labels[si,:]

        gt_label_str = get_symbols_from_codes(gt_label)
        pr_label_str = get_symbols_from_codes(pr_label)

        corr_mask = gt_label == pr_label

        if only_error and corr_mask.all():
            continue

        dst_h = cfg.INPUT_SHAPE[1]
        fy = dst_h / float(vis_img.shape[0])
        fx = fy

        vis_img = convert_color_space(vis_img, cfg, inverse=True)
        vis_img = cv2.resize(vis_img, (0, 0), fx=fx, fy=fy)

        imname = '[gt]{}_[pr]{}_[im]{}'.format(gt_label_str, pr_label_str, imname_orig)
        cv2.imwrite(join(output_dir, imname), vis_img)


def test_net(net, model_id, test_cfg, cfg):

    datasets = test_cfg.DATASETS

    for indx, dataset in enumerate(datasets):

        output_dir = None
        prepared = False
        do_visualize = False

        if 'OUTPUT_DIR' in dataset:
            output_dir = dataset['OUTPUT_DIR']

            if 'APPEND_EPOCH' in dataset:
                output_dir_n = output_dir[:-1] if output_dir.endswith('/') else output_dir
                output_dir_last = basename(output_dir_n)
                output_base_dir = dirname(output_dir_n)
                output_dir = join(output_base_dir, '{}_{}'.format(output_dir_last, model_id))
                print(output_dir)

            if 'VISUALIZE' in dataset:
                do_visualize = dataset['VISUALIZE']

            if not isdir(output_dir):
                makedirs(output_dir)

        if 'PREPARED' in dataset:
            prepared = True

        test_dataset = CollectionDataset([dataset], cfg, test_cfg=test_cfg, output_idx=True, prepared=prepared)
        test_iter = MxIterWrapper(test_dataset, cfg, test_cfg=test_cfg, num_workers=cfg.TEST.LOADER_WORKERS)
        net.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label)

        print('[{}/{}] dataset {}: {} samples'.format(indx + 1, len(datasets), dataset.DB_PATH, len(test_iter)))

        eval_metric = CaptchaAccuracy(cfg)

        test_iter.reset()
        eval_metric.reset()

        gt_labels_all = None
        pr_labels_all = None
        imnames_all = None

        n_samples = len(test_iter)
        n_batches = int(n_samples / test_iter.batch_size)
        n_batches += 1 if n_samples % test_iter.batch_size > 0 else 0
        print('number of iterations: {}'.format(n_batches))

        with tqdm(total=n_batches) as progress_bar:
            for nbatch, iter_data in enumerate(test_iter):
                test_batch, samples_idx = iter_data
                samples_idx = list(samples_idx.asnumpy())
                imnames = [basename(test_dataset.get_imname(sid)) for sid in samples_idx]
                net.forward(test_batch, is_train=False)

                preds = net.get_outputs()
                labels = test_batch.label
                batch_size = len(samples_idx)

                eval_metric.update(labels=labels, preds=preds)

                pr_labels = np.zeros((batch_size, len(preds)), dtype=np.int32)
                for i in range(len(preds)):
                    prob_i = preds[i].asnumpy()
                    maxprob_arg_i = np.argmax(prob_i, axis=1)
                    pr_labels[:, i] = maxprob_arg_i.astype(np.int32)

                gt_labels = np.zeros((batch_size, len(labels)), dtype=np.int32)
                for i in range(len(labels)):
                    curr_labels = labels[i].asnumpy()
                    gt_labels[:, i] = curr_labels.astype(np.int32)

                gt_labels_all = gt_labels if gt_labels_all is None else np.concatenate((gt_labels_all, gt_labels))
                pr_labels_all = pr_labels if pr_labels_all is None else np.concatenate((pr_labels_all, pr_labels))
                imnames_all = imnames if imnames_all is None else imnames_all + imnames

                # visualize results
                if output_dir is not None and do_visualize:
                    imgs = test_batch.data[0]
                    imgs = mx.nd.transpose(imgs, axes=(0, 2, 3, 1))
                    imgs = imgs.asnumpy()

                    visualize_result(imgs, imnames, gt_labels, pr_labels, output_dir, cfg)

                progress_bar.update(1)

        print('test results:')
        for name, val in eval_metric.get_name_value():
            print('{}: {:.5f}'.format(name, val))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--gpus', dest='gpu_ids',
                        help='GPU device id to use 0,1,2', type=str)
    parser.add_argument('--exp-name', dest='exp_name', required=True, type=str)
    parser.add_argument('--model', dest='model', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    model_name = basename(args.model)
    exp_name = args.exp_name
    exp_dir = join('exps', exp_name)

    snapshots_dir = join(exp_dir, 'snapshots')

    cfg_file = join(exp_dir, 'config.yml')
    cfg_from_file(cfg_file)

    if args.gpu_ids:
        cfg.GPU_IDS = [int(gpu) for gpu in args.gpu_ids.split(',')]

    if len(cfg.GPU_IDS) == 0:
        # use all available GPU
        print('use all GPU')
        devices = [mx.gpu(i) for i in range(cfg.GPU_NUM)]
    else:
        print('use GPU: {}'.format(cfg.GPU_IDS))
        devices = [mx.gpu(i) for i in cfg.GPU_IDS]

    np.random.seed(cfg.RNG_SEED)
    mx.random.seed(cfg.RNG_SEED)

    prefix = join(snapshots_dir, model_name[:-12])
    checkpoint_epoch = int(model_name[-11:-7])
    model_id = splitext(model_name)[0]

    label_names = ['softmax{}_label'.format(i) for i in range(cfg.MAX_LABELS)]
    net = mx.mod.Module.load(prefix, checkpoint_epoch, label_names=label_names, context=devices)
    test_net(net, model_id, cfg.TEST, cfg)