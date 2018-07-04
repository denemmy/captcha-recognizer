import numpy as np
from utils import crop_image, expand_bbox, resize_image, prepare_crop
import cv2
import mxnet as mx

class SampleWithCache:
    """with cache
    """

    _CACHE_MAX_SIZE = 0 * 1024 * 1024 * 1024 # 16 Gb
    _CACHE = {}
    _CACHE_SIZE = 0

    def __init__(self, img_path, annotation, cfg):
        self._img_path = img_path

        if cfg.INPUT_FORMAT.FORMAT == 'labels':
            labels = [int(l) for l in annotation]
            self._labels = labels
        else:
            raise NotImplementedError

        self._argmax_label = -1
        self._cfg = cfg

    def get_data(self):
        if self._img_path in SampleWithCache._CACHE:
            im = SampleWithCache._CACHE[self._img_path]
        else:
            im = cv2.imread(self._img_path)
            im = im[:,:,[2, 1, 0]]
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # !! <- deadlock when python multiprocessing is used

            # prepare size
            assert self._cfg.PREPARE_SCALE is not None or self._cfg.PREPARE_SIZE is not None
            if self._cfg.PREPARE_SCALE is not None:
                im = resize_image(im, scale=self._cfg.PREPARE_SCALE)
            else:
                prepare_sz = tuple(self._cfg.PREPARE_SIZE)
                im = prepare_crop(im, prepare_sz)

            if SampleWithCache._CACHE_SIZE < SampleWithCache._CACHE_MAX_SIZE:
                SampleWithCache._CACHE[self._img_path] = im
                SampleWithCache._CACHE_SIZE += im.shape[0] * im.shape[1] * im.shape[2] * 1 # in bytes (type uint8)

        return im.copy()

    def get_label(self):
        return self._labels

    @property
    def id(self):
        return str(self._img_path)

    @staticmethod
    def reset_cache():
        SampleWithCache._CACHE = {}
        SampleWithCache._CACHE_SIZE = 0

    @staticmethod
    def set_cache_size(max_gb):
        SampleWithCache._CACHE_MAX_SIZE = int(max_gb * 1024 * 1024 * 1024)