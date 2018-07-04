import numpy as np
import cv2
import mxnet as mx

def prepare_crop(im, prepare_sz):
    if im.shape[0] != prepare_sz[1] or im.shape[1] != prepare_sz[0]:
        prepare_r = float(prepare_sz[0]) / prepare_sz[1]
        orig_r = float(im.shape[1]) / im.shape[0]

        if orig_r < prepare_r:
            # fit width
            crop_w = im.shape[1]
            crop_h = crop_w / prepare_r
        else:
            # fit height
            crop_h = im.shape[0]
            crop_w = crop_h * prepare_r

        crop_x = int((im.shape[1] - crop_w) / 2.)
        crop_y = int((im.shape[0] - crop_h) / 2.)
        crop_w = int(crop_w)
        crop_h = int(crop_h)

        im = im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]

        interp = cv2.INTER_AREA if im.shape[1] > prepare_sz[0] else cv2.INTER_LINEAR
        im = cv2.resize(im, prepare_sz, interpolation=interp)
    return im

def transform(im, prepare_sz, target_shape):

    target_shape = tuple(target_shape)
    im = im.astype(np.float32, copy=False)

    prepare_sz = tuple(prepare_sz)
    im = prepare_crop(im, prepare_sz)

    # crop
    im_w = im.shape[1]
    im_h = im.shape[0]

    # scale crop
    crop_shape = (target_shape[0], target_shape[1])

    # check max side
    if crop_shape[0] > im_w or crop_shape[1] > im_h:
        crop_r = float(crop_shape[0]) / crop_shape[1]
        img_r = float(im_w) / im_h
        if crop_r < img_r:
            # fit height
            crop_shape = (int(im_h * crop_r), im_h)
        else:
            # fit width
            crop_shape = (im_w, int(im_w / crop_r))

    cr = [0, 0, crop_shape[0], crop_shape[1]]
    cr[0] = max(int((im_w - crop_shape[0]) * 0.5 + 0.5), 0)
    cr[1] = max(int((im_h - crop_shape[1]) * 0.5 + 0.5), 0)
    im = im[cr[1]:cr[1]+cr[3],cr[0]:cr[0]+cr[2],:]

    # fit to target shape if needed
    if im.shape[0] != target_shape[1] or im.shape[1] != target_shape[0]:
        im = cv2.resize(im, target_shape, interpolation=cv2.INTER_AREA)

    return im