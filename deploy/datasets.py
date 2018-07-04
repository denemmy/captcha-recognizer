from os.path import join, splitext, isfile
from os import listdir
from mxnet.gluon.data import dataset
from transform import transform
import numpy as np
import mxnet as mx
import cv2
from collections import namedtuple

def list_images(base_dir, valid_exts=['.jpg', '.jpeg', '.png', '.bmp', '.ppm']):
    images_list = []
    for f in listdir(base_dir):
        if not isfile(join(base_dir, f)):
            continue
        filext = splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list

class Sample:
    def __init__(self, img_path):
        self._img_path = img_path

    def get_data(self):
        im = cv2.imread(self._img_path)
        im = im[:,:,[2, 1, 0]]
        return im.copy()

    @property
    def id(self):
        return str(self._img_path)

class CollectionDataset(dataset.Dataset):

    def __init__(self, input_dir, prepare_sz, target_shape):
        self._input_dir = input_dir
        self._samples = self.load_samples()
        self._prepare_sz = prepare_sz
        self._target_shape = target_shape

    def load_samples(self):
        imnames = list_images(self._input_dir)
        return [Sample(join(self._input_dir, imname)) for imname in imnames]

    def __getitem__(self, idx):
        sample = self._samples[idx]
        img = sample.get_data()
        img = transform(img, self._prepare_sz, self._target_shape)
        channel_swap = (2, 0, 1)
        img = np.transpose(img, axes=channel_swap)

        return img, np.int32(idx)

    def get_imname(self, idx):
        return self._samples[idx].id

    def __len__(self):
        return len(self._samples)