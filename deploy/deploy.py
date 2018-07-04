import sys
import numpy as np
import mxnet as mx
from os import mkdir, makedirs
from os.path import join, isdir, basename, dirname, splitext
import argparse
import cv2
import tqdm

from datasets import CollectionDataset

INPUT_SHAPE = (80, 35)
PREPARE_SIZE = (88, 40)
BATCH_SIZE = 64

def code_to_symbol(code):
    if code < 10:
        return chr(ord('0') + code)
    else:
        return chr(ord('A') + code - 10)

def get_symbols_from_codes(codes):
    return ''.join([code_to_symbol(code) for code in codes])

def test_net(net, ctx, input_dir):

    test_dataset = CollectionDataset(input_dir, PREPARE_SIZE, INPUT_SHAPE)
    if len(test_dataset) == 0:
        print('no images found')
        return

    print('loaded {} samples'.format(len(test_dataset)))
    test_dataloader = mx.gluon.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=0)
    test_iter = iter(test_dataloader)
    result = []

    n_samples = len(test_dataset)
    n_batches = int(n_samples / BATCH_SIZE)
    n_batches += 1 if n_samples % BATCH_SIZE > 0 else 0
    print('number of iterations: {}'.format(n_batches))

    with tqdm.tqdm(total=n_batches) as progress_bar:
        for nbatch, iter_data in enumerate(test_iter):
            data, samples_idx = iter_data
            samples_idx = list(samples_idx.asnumpy())
            imnames = [basename(test_dataset.get_imname(sid)) for sid in samples_idx]
            data = data.as_in_context(ctx[0])
            preds = net(data)

            batch_size = len(samples_idx)

            pr_labels = np.zeros((batch_size, len(preds)), dtype=np.int32)
            for i in range(len(preds)):
                prob_i = preds[i].asnumpy()
                maxprob_arg_i = np.argmax(prob_i, axis=1)
                pr_labels[:, i] = maxprob_arg_i.astype(np.int32)

            batch_labels = []
            for i in range(pr_labels.shape[0]):
                labels = get_symbols_from_codes(list(pr_labels[i,:]))
                batch_labels.append(labels)
            result.extend(zip(imnames, batch_labels))
            progress_bar.update(1)

    with open(join(input_dir, 'results.txt'), 'w') as fp:
        for imname, label in result:
            fp.write('{};{}\n'.format(imname, label))

    print('all done.')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--input', dest='input_dir',
                        help='input directory with images', type=str, required=True)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use', type=int, default=0)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    ctx = [mx.gpu(args.gpu_id)]
    input_dir = args.input_dir

    prefix = 'model/model-best'
    checkpoint_epoch = 0

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, checkpoint_epoch)
    all_layers = sym.get_internals()
    sym = mx.sym.Group([all_layers['fc_softmax{}_output'.format(i)] for i in range(5)])

    net = mx.gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))
    # Set the params
    net_params = net.collect_params()
    for param in arg_params:
        if param in net_params:
            net_params[param]._load_init(arg_params[param], ctx=ctx)
    for param in aux_params:
        if param in net_params:
            net_params[param]._load_init(aux_params[param], ctx=ctx)

    test_net(net, ctx, input_dir)