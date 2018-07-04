from os.path import isdir, join, basename
from mxnet import ndarray as nd
from os import mkdir
import numpy as np
import mxnet as mx
import random
import cv2
from loaders import load_samples
from transform import transform, convert_color_space

class MxIterWrapper(mx.io.DataIter):

    def __init__(self, input_dataset, cfg, test_cfg=None,
                 data_name='data', label_name='softmax', num_workers=0):
        self._cfg = cfg
        self._test_cfg = test_cfg
        self._is_train = test_cfg is None
        self._samples_num = len(input_dataset)

        self._batch_size = cfg.TRAIN.BATCH_SIZE if self._is_train else test_cfg.BATCH_SIZE
        self._debug_imgs = cfg.TRAIN.DEBUG_IMAGES if self._is_train else test_cfg.DEBUG_IMAGES
        self._channel_num = 1 if cfg.GRAYSCALE else 3
        self._data_shape = (self._channel_num, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0])

        self._provide_data = [(data_name, (self._batch_size,) + self._data_shape)]

        self._provide_label = [('{}{}_label'.format(label_name, i),
                                (self._batch_size,)) for i in range(cfg.MAX_LABELS)]

        self._input_dataset = input_dataset
        self._num_workers = num_workers

        self._train_data = mx.gluon.data.DataLoader(input_dataset, batch_size=self._batch_size,
                                                    shuffle=self._is_train, num_workers=num_workers)
        self._train_iter = iter(self._train_data)

    def reset(self):
        self._train_iter = iter(self._train_data)

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def next(self):
        batch = next(self._train_iter)

        assert len(batch) in (2, 3)
        batch_data = batch[0]
        batch_label = batch[1]
        batch_idx = batch[2] if len(batch) == 3 else None

        assert batch_data.shape[0] == batch_label.shape[0]
        n_samples = batch_data.shape[0]

        batch_label_lst = []
        for i in range(batch_label.shape[1]):
            batch_label_lst.append(batch_label[:, i])

        data_batch = mx.io.DataBatch([batch_data], batch_label_lst, pad=(self._batch_size - n_samples))

        if batch_idx is None:
            return data_batch
        else:
            return data_batch, batch_idx

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return self._samples_num