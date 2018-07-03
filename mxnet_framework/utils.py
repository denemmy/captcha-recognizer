import numpy as np
from os import listdir
from os.path import join, splitext, isfile
import cv2

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

def crop_image(img, bbox):
    x_st = bbox[0]
    y_st = bbox[1]

    x_en = bbox[0] + bbox[2] - 1
    y_en = bbox[1] + bbox[3] - 1

    x_st_pad = int(max(0, -x_st))
    y_st_pad = int(max(0, -y_st))
    x_en_pad = int(max(0, x_en - img.shape[1] + 1))
    y_en_pad = int(max(0, y_en - img.shape[0] + 1))

    x_en = x_en + max(0, -x_st)
    y_en = y_en + max(0, -y_st)
    x_st = max(0, x_st)
    y_st = max(0, y_st)

    if y_st_pad != 0 or y_en_pad != 0 or x_st_pad != 0 or x_en_pad != 0:
        assert len(img.shape) in (2, 3)
        if len(img.shape) == 3:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad, img.shape[2]), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1], :] = img
        else:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1]] = img
    else:
        img_pad = img
    img_cropped = img_pad[y_st:y_en+1, x_st:x_en+1]
    return img_cropped

def expand_bbox(bbox, scale):
    bbox_c_x = bbox[0] + (bbox[2] - 1) / 2.0
    bbox_c_y = bbox[1] + (bbox[3] - 1) / 2.0

    bbox_max_side = max(bbox[2], bbox[3])
    bbox_new_size = scale * bbox_max_side

    bbox[0] = int(bbox_c_x - (bbox_new_size - 1) / 2.0)
    bbox[1] = int(bbox_c_y - (bbox_new_size - 1) / 2.0)

    bbox[2] = int(bbox_new_size)
    bbox[3] = int(bbox_new_size)
    return bbox

def resize_image(im, scale=240):
    min_side = min(im.shape[:2])
    if min_side == scale:
        return im

    target_r = scale / float(min_side)

    interp = cv2.INTER_AREA if target_r < 1 else cv2.INTER_LINEAR
    im = cv2.resize(im, None, fx=target_r, fy=target_r, interpolation=interp)

    return im

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



