from mxnet.gluon.data import dataset
from os.path import join, isdir, basename
from os import mkdir
import cv2
import numpy as np
import mxnet as mx
import time
from transform import transform
from loaders import load_samples
from transform import convert_color_space

class CollectionDataset(dataset.Dataset):

    def __init__(self, db_params, cfg, test_cfg=None, prepared=False, output_idx=False):
        self._cfg = cfg
        self._test_cfg = test_cfg
        self._is_train = test_cfg is None
        self._prepared = prepared
        self._output_idx = output_idx

        self._n_debug_images = 0
        self._max_debug_images = cfg.MAX_DEBUG_IMAGES
        self._csv_workers = cfg.TRAIN.CSV_PARSER_WORKERS if self._is_train else test_cfg.CSV_PARSER_WORKERS

        # make debug directory
        self._debug_imgs = cfg.TRAIN.DEBUG_IMAGES if self._is_train else test_cfg.DEBUG_IMAGES
        if self._debug_imgs:
            if not isdir(self._debug_imgs):
                mkdir(self._debug_imgs)

        print('loading csv (number of workers: {})..'.format(self._csv_workers))
        tic = time.time()
        self._samples = load_samples(db_params, self._cfg, self._csv_workers)
        self._filter_samples()
        print('parsing finished in {:.3} sec'.format(time.time() - tic))

    def _filter_samples(self):
        cfg = self._cfg
        samples = self._samples
        n_before_filt = len(samples)
        # filter
        samples = [s for s in samples if len(s.get_label()) >= cfg.MIN_LABELS and len(s.get_label()) <= cfg.MAX_LABELS]
        n_after_filr = len(samples)
        print('checking samples labels, filter: {} --> {}'.format(n_before_filt, n_after_filr))
        self._samples = samples

    def __getitem__(self, idx):
        sample = self._samples[idx]
        img = sample.get_data()
        labels_orig = sample.get_label()
        n_orig = len(labels_orig)
        labels = list(labels_orig)

        # if n_orig < self._cfg.MAX_LABELS:
        #     labels = labels_orig + [-1] * (self._cfg.MAX_LABELS - n_orig) + [n_orig - self._cfg.MIN_LABELS]
        # else:
        #     labels = labels_orig + [n_orig - self._cfg.MIN_LABELS]

        img = transform(img, self._cfg, self._is_train, not self._prepared)
        if self._debug_imgs:
            self._debug_image(img, labels_orig, sample.id)

        channel_swap = (2, 0, 1)
        img = np.transpose(img, axes=channel_swap)

        if self._output_idx:
            return img, np.float32(labels), np.int32(idx)
        else:
            return img, np.float32(labels)

    def get_imname(self, idx):
        return self._samples[idx].id

    def _debug_image(self, im, label, imname):

        if self._n_debug_images > self._max_debug_images:
            return

        # color space
        im = convert_color_space(im, self._cfg, inverse=True)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        label_str = '.'.join([str(l) for l in label])
        imname_n = '{}_{}'.format(label_str, basename(imname))
        cv2.imwrite(join(self._debug_imgs, imname_n), im)
        self._n_debug_images += 1

    def __len__(self):
        return len(self._samples)


