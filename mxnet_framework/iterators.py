from os.path import isdir, join, basename
from mxnet import ndarray as nd
from os import mkdir
import numpy as np
import mxnet as mx
import random
import cv2
from loaders import load_samples
from transform import transform_test_multi_crop, transform, convert_color_space

class ClsIter(mx.io.DataIter):

    def __init__(self, db_params, cfg, data_name='data',
                 label_name='label', prepared=False, test_cfg=None, output_imnames=False):

        self._cfg = cfg
        self._test_cfg = test_cfg
        self._is_train = test_cfg is None

        self._csv_workers = cfg.TRAIN.CSV_PARSER_WORKERS if self._is_train else test_cfg.CSV_PARSER_WORKERS

        self._output_imnames = output_imnames
        self._prepared = prepared

        self._max_debug_images = cfg.MAX_DEBUG_IMAGES
        self._n_debug_images = 0

        self._batch_size = cfg.TRAIN.BATCH_SIZE if self._is_train else test_cfg.BATCH_SIZE
        self._multi_crop = False if self._is_train else test_cfg.MULTI_CROP
        self._channel_num = 1 if cfg.GRAYSCALE else 3
        self._data_shape = (self._channel_num, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0])

        self._provide_data = [(data_name, (self._batch_size,) + self._data_shape)]
        self._provide_label = [(label_name, (self._batch_size, cfg.MAX_LABELS + 1))]

        # make debug directory
        self._debug_imgs = cfg.TRAIN.DEBUG_IMAGES if self._is_train else test_cfg.DEBUG_IMAGES
        if self._debug_imgs:
            if not isdir(self._debug_imgs):
                mkdir(self._debug_imgs)

        # load samples
        self._samples = load_samples(db_params, self._cfg, self._csv_workers)

        self._init_iter_once()
        self._init_iter()

    def _init_iter_once(self):

        self._indx = 0
        self._samples_num = len(self._samples)
        self._order = list(range(self._samples_num))

    def _init_iter(self):
        self._indx = 0
        self._n_batch = 0
        if self._is_train:
            random.shuffle(self._order)

    def __iter__(self):
        return self

    def reset(self):
        self._init_iter()

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._samples_num if not self._multi_crop else 2 * self._samples_num

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    @property
    def batch_size(self):
        return self._batch_size

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self._indx >= self._samples_num:
            raise StopIteration()
        else:
            sample = self._samples[self._order[self._indx]]
            self._indx += 1

        return sample

    def next(self):
        """Returns the next batch of data."""
        batch_size = self._batch_size
        c, h, w = self._data_shape

        multi_crop = self._multi_crop

        batch_data = np.zeros((batch_size, h, w, c))
        batch_label = np.zeros((batch_size, self._cfg.MAX_LABELS + 1))

        imnames = []
        sample_idx = 0
        try:
            while sample_idx < batch_size:
                sample = self.next_sample()
                im = sample.get_data()
                labels_orig = sample.get_label()
                n_orig = len(labels_orig)
                if n_orig < self._cfg.MAX_LABELS:
                    labels = labels_orig + [-1] * (self._cfg.MAX_LABELS - n_orig) + [n_orig]
                else:
                    labels = labels_orig + [n_orig]

                imname = basename(sample.id)

                if multi_crop:
                    imgs = transform_test_multi_crop(im, self._cfg)
                    assert len(imgs) < batch_size
                    remain_to_add = batch_size - sample_idx
                    if len(imgs) > remain_to_add:
                        raise StopIteration

                    for s_idx, im in enumerate(imgs):
                        if self._debug_imgs:
                            imname_o = '{}_{}'.format(s_idx, imname)
                            self._debug_image(im, labels_orig, imname_o)

                        batch_data[sample_idx] = im
                        batch_label[sample_idx] = labels
                        imnames.append(imname)

                        sample_idx += 1
                else:
                    im = transform(im, self._cfg, self._is_train, not self._prepared)

                    # debug images
                    imnames.append(imname)
                    if self._debug_imgs:
                        self._debug_image(im, labels_orig, imname)

                    batch_data[sample_idx] = im
                    batch_label[sample_idx] = labels

                    sample_idx += 1
        except StopIteration:
            if not sample_idx:
                raise StopIteration
            batch_data = batch_data[:sample_idx,:,:,:]
            batch_label = batch_label[:sample_idx,:]

        channel_swap = (0, 3, 1, 2)
        batch_data = np.transpose(batch_data, axes=channel_swap)

        batch_data = nd.array(batch_data)
        batch_label = nd.array(batch_label)

        data_batch = mx.io.DataBatch([batch_data], [batch_label], pad=(batch_size-sample_idx))
        self._n_batch += 1

        # output imnames for visualizing
        if self._output_imnames:
            return data_batch, imnames
        else:
            return data_batch

    def _debug_image(self, im, label, imname):

        if self._n_debug_images > self._max_debug_images:
            return

        # color space
        im = convert_color_space(im, self._cfg, inverse=True)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # imname_n = '{}_{}'.format(label, basename(imname))
        cv2.imwrite(join(self._debug_imgs, imname), im)

        self._n_debug_images += 1

class MxIterWrapper(mx.io.DataIter):

    def __init__(self, input_dataset, cfg, test_cfg=None,
                 data_name='data', label_name='label', num_workers=0):
        self._cfg = cfg
        self._test_cfg = test_cfg
        self._is_train = test_cfg is None
        self._samples_num = len(input_dataset)

        self._batch_size = cfg.TRAIN.BATCH_SIZE if self._is_train else test_cfg.BATCH_SIZE
        self._debug_imgs = cfg.TRAIN.DEBUG_IMAGES if self._is_train else test_cfg.DEBUG_IMAGES
        self._channel_num = 1 if cfg.GRAYSCALE else 3
        self._data_shape = (self._channel_num, cfg.INPUT_SHAPE[1], cfg.INPUT_SHAPE[0])

        self._provide_data = [(data_name, (self._batch_size,) + self._data_shape)]
        self._provide_label = [(label_name, (self._batch_size, cfg.MAX_LABELS + 1))]

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

        data_batch = mx.io.DataBatch([batch_data], [batch_label], pad=(self._batch_size - n_samples))

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