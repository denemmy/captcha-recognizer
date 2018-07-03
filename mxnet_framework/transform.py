import numpy as np
from utils import resize_image, prepare_crop
import cv2
import mxnet as mx

def convert_color_space(im, cfg, inverse=False):
    if not inverse:
        if cfg.SUBTR_MEANS:
            if cfg.MEANS_TYPE == 'pixel':
                pxmeans = np.array(cfg.PIXEL_MEANS)
                pxstd = np.array(cfg.PIXEL_STD)
                im -= pxmeans
                im /= pxstd
            elif cfg.MEANS_TYPE == 'image':
                im -= cfg.MEAN_IMG
                im /= cfg.STD_IMG

        if cfg.GRAYSCALE:
            im = im * cfg.NORM_COEFF if cfg.NORM_COEFF != 1.0 else im
            if not cfg.USE_IMADJUST:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
        else:
            im = im * cfg.NORM_COEFF if cfg.NORM_COEFF != 1.0 else im
            if cfg.COLOR_SPACE == 'bgr':
                # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) # !! <- deadlock when python multiprocessing is used
                im = im[:, :, [2, 1, 0]]
    else:
        if cfg.GRAYSCALE:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            im = im / cfg.NORM_COEFF if cfg.NORM_COEFF != 1.0 else im
        else:
            if cfg.COLOR_SPACE == 'bgr':
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # !! <- deadlock when python multiprocessing is used
                im = im[:,:,[2, 1, 0]]
            im = im / cfg.NORM_COEFF if cfg.NORM_COEFF != 1.0 else im
        if cfg.SUBTR_MEANS:
            if cfg.MEANS_TYPE == 'pixel':
                pxmeans = np.array(cfg.PIXEL_MEANS)
                pxstd = np.array(cfg.PIXEL_STD)
                im *= pxstd
                im += pxmeans
            elif cfg.MEANS_TYPE == 'image':
                im *= cfg.STD_IMG
                im += cfg.MEAN_IMG
    return im

def transform_test_multi_crop(im, cfg):
    target_shape = tuple(cfg.INPUT_SHAPE)

    im = im.astype(np.float32, copy=False)

    imgs = []

    # prepare size
    assert cfg.PREPARE_SCALE is not None or cfg.PREPARE_SIZE is not None
    if cfg.PREPARE_SCALE is not None:
        im = resize_image(im, scale=cfg.PREPARE_SCALE)
    else:
        prepare_sz = tuple(cfg.PREPARE_SIZE)
        im = prepare_crop(im, prepare_sz)

    im_w = im.shape[1]
    im_h = im.shape[0]

    crop_shape = (target_shape[0], target_shape[1])
    cr = [0, 0, crop_shape[0], crop_shape[1]]
    cr[0] = max(int((im_w - crop_shape[0]) * 0.5 + 0.5), 0)
    cr[1] = max(int((im_h - crop_shape[1]) * 0.5 + 0.5), 0)

    im_1 = im[cr[1]:cr[1] + cr[3], cr[0]:cr[0] + cr[2], :]
    im_2 = np.fliplr(im_1)

    imgs.append(im_1)
    imgs.append(im_2)

    # scaled to max and its flip
    crop_shape = (target_shape[0], target_shape[1])
    max_scale = cfg.TRAIN.DISTORT.SCALE_MAX
    ratio = max_scale
    crop_shape = (int(crop_shape[0] * ratio), int(crop_shape[1] * ratio))

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
    im_3 = im[cr[1]:cr[1] + cr[3], cr[0]:cr[0] + cr[2], :]
    im_4 = np.fliplr(im_3)

    # imgs.append(im_3)
    # imgs.append(im_4)

    # moved crop
    crop_shape = (target_shape[0], target_shape[1])
    max_side = max([im_w, im_h])
    min_side = max([im_w, im_h])
    ratio = float(max_side) / min_side
    if ratio > 1.2:

        ratio = 0.8

        img_r = float(im_w) / im_h
        if img_r > 1:
            # fit height
            ratio_x = ratio
            ratio_y = 0
        else:
            # fit width
            ratio_x = 0
            ratio_y = ratio

        # rectangle where image is cropped
        crop_rect_w = int(crop_shape[0] + ratio_x * (im_w - crop_shape[0]))
        crop_rect_h = int(crop_shape[1] + ratio_y * (im_h - crop_shape[1]))

        dif_w = min(im_w, crop_rect_w) - crop_shape[0]
        dif_h = min(im_h, crop_rect_h) - crop_shape[1]

        # shift + center
        cr[0] = 0 + max(int((im_w - crop_rect_w) * 0.5 + 0.5), 0)
        cr[1] = 0 + max(int((im_h - crop_rect_h) * 0.5 + 0.5), 0)
        im_5 = im[cr[1]:cr[1] + cr[3], cr[0]:cr[0] + cr[2], :]

        cr[0] = dif_w + max(int((im_w - crop_rect_w) * 0.5 + 0.5), 0)
        cr[1] = dif_h + max(int((im_h - crop_rect_h) * 0.5 + 0.5), 0)
        im_6 = im[cr[1]:cr[1] + cr[3], cr[0]:cr[0] + cr[2], :]

        # imgs.append(im_5)
        # imgs.append(im_6)

    imgs_o = []
    for img_i in imgs:
        # fit to target shape if needed
        if img_i.shape[0] != target_shape[1] or img_i.shape[1] != target_shape[0]:
            img_i = cv2.resize(img_i, target_shape, interpolation=cv2.INTER_AREA)

        img_i = convert_color_space(img_i, cfg, inverse=False)
        imgs_o.append(img_i)

    return imgs_o

def transform(im, cfg, do_aug, do_crop):

    target_shape = tuple(cfg.INPUT_SHAPE)

    im = im.astype(np.float32, copy=False)

    # prepare size
    assert cfg.PREPARE_SCALE is not None or cfg.PREPARE_SIZE is not None
    if cfg.PREPARE_SCALE is not None:
        im = resize_image(im, scale=cfg.PREPARE_SCALE)
    else:
        prepare_sz = tuple(cfg.PREPARE_SIZE)
        im = prepare_crop(im, prepare_sz)

    # pad
    if do_aug and cfg.TRAIN.DISTORT.USE_PAD:
        pad_size = cfg.TRAIN.DISTORT.PAD_SIZE
        im = cv2.copyMakeBorder(im, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    # deformation
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.DEFORM_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_DEFORM and to_go:
        # choose axis, 50% x-axis, 50% y-axis
        ratio = np.random.uniform(1., cfg.TRAIN.DISTORT.MAX_DEFORM)
        if np.random.randint(0, 2):
            # x axis
            im = cv2.resize(im, (0, 0), fx=ratio, fy=1., interpolation=cv2.INTER_LINEAR)
        else:
            # y axis
            im = cv2.resize(im, (0, 0), fx=1., fy=ratio, interpolation=cv2.INTER_LINEAR)

    # rotate
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.ROTATE_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_ROTATE and to_go:
        im_w = im.shape[1]
        im_h = im.shape[0]
        angle = np.random.uniform(-cfg.TRAIN.DISTORT.MAX_ROTATE, cfg.TRAIN.DISTORT.MAX_ROTATE)
        M = cv2.getRotationMatrix2D((im_w / 2. + 0.5, im_h / 2. + 0.5), angle, 1)
        im = cv2.warpAffine(im, M, (im_w, im_h))

    # crop
    im_w = im.shape[1]
    im_h = im.shape[0]

    # scale crop
    crop_shape = (target_shape[0], target_shape[1])
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.SCALE_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_SCALE and to_go:
        min_scale = cfg.TRAIN.DISTORT.SCALE_MIN
        max_scale = cfg.TRAIN.DISTORT.SCALE_MAX
        ratio = np.random.uniform(min_scale, max_scale)
        crop_shape = (int(crop_shape[0] * ratio), int(crop_shape[1] * ratio))

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

    # move crop
    if do_aug and cfg.TRAIN.DISTORT.USE_CROP and np.random.randint(0, 2):

        ratio = cfg.TRAIN.DISTORT.CROP_REGION
        ratio = max(0., ratio)
        ratio = min(1., ratio)

        # rectangle where image is cropped
        crop_rect_w = int(crop_shape[0] + ratio * (im_w - crop_shape[0]))
        crop_rect_h = int(crop_shape[1] + ratio * (im_h - crop_shape[1]))

        dif_w = min(im_w, crop_rect_w) - crop_shape[0]
        dif_h = min(im_h, crop_rect_h) - crop_shape[1]

        rand_off_x = np.random.randint(0, dif_w+1)
        rand_off_y = np.random.randint(0, dif_h+1)

        cr[0] = rand_off_x + max(int((im_w - crop_rect_w) * 0.5 + 0.5), 0)
        cr[1] = rand_off_y + max(int((im_h - crop_rect_h) * 0.5 + 0.5), 0)

    if do_crop:
        im = im[cr[1]:cr[1]+cr[3],cr[0]:cr[0]+cr[2],:]

    # mirror
    if do_aug and cfg.TRAIN.DISTORT.USE_FLIP and np.random.randint(0, 2):
        im = np.fliplr(im)

    # jpeg:
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.JPEG_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_JPEG and to_go:
        # compress
        jpeg_min_q = cfg.TRAIN.DISTORT.JPEG_MIN_QUALITY
        jpeg_max_q = cfg.TRAIN.DISTORT.JPEG_MAX_QUALITY

        jpeg_quality = np.random.uniform(jpeg_min_q, jpeg_max_q)
        im_code = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
        # uncompress
        im = cv2.imdecode(im_code, -1)

        im = im.astype(np.float32)

    # noise
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.NOISE_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_NOISE and to_go:
        sigma = np.random.uniform(0, cfg.TRAIN.DISTORT.NOISE_SIGMA)
        noise_to_add = sigma * 255 * np.random.randn(im.shape[0], im.shape[1], im.shape[2])
        noise_to_add = noise_to_add.astype(np.float32)
        im = im + noise_to_add

    # low resolution
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.LOW_RES_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_LOW_RES and to_go:
        factor = np.random.uniform(cfg.TRAIN.DISTORT.LOW_RES_FACTOR_MIN, cfg.TRAIN.DISTORT.LOW_RES_FACTOR_MAX)
        im_shape = (im.shape[1], im.shape[0]) # (x, y)
        low_shape = (int(im_shape[0] / float(factor)), int(im_shape[1] / float(factor)))
        im = cv2.resize(im, low_shape, interpolation=cv2.INTER_AREA)
        im = cv2.resize(im, im_shape, interpolation=cv2.INTER_LINEAR)

    # brightness:
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.BRIGHTNESS_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_BRIGHTNESS and to_go:

        br_min_a = cfg.TRAIN.DISTORT.BRIGHT_MIN_ALPHA
        br_max_a = cfg.TRAIN.DISTORT.BRIGHT_MAX_ALPHA

        br_min_b = cfg.TRAIN.DISTORT.BRIGHT_MIN_BETA
        br_max_b = cfg.TRAIN.DISTORT.BRIGHT_MAX_BETA

        alpha = np.random.uniform(br_min_a, br_max_a)
        beta = np.random.uniform(br_min_b, br_max_b)
        im = im * alpha + beta

        im = cv2.threshold(im, 255, 255, cv2.THRESH_TRUNC)[1]
        im = cv2.threshold(im, 0, 0, cv2.THRESH_TOZERO)[1]

    # white balance
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.WHITE_BALANCE_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_WHITE_BALANCE and to_go:

        im_b = im[:, :, 0]
        im_g = im[:, :, 1]
        im_r = im[:, :, 2]

        max_coeff = cfg.TRAIN.DISTORT.WHITE_BALANCE_COEFF
        coeffs = 1 + np.random.uniform(-max_coeff, max_coeff, 3)

        im_b *= coeffs[0]
        im_g *= coeffs[1]
        im_r *= coeffs[2]

        im[:, :, 0] = im_b
        im[:, :, 1] = im_g
        im[:, :, 2] = im_r

        # check bounds
        im = cv2.threshold(im, 255, 255, cv2.THRESH_TRUNC)[1]
        im = cv2.threshold(im, 0, 0, cv2.THRESH_TOZERO)[1]

    # grayscale
    to_go = np.random.uniform(0, 1) < cfg.TRAIN.DISTORT.GRAYSCALE_PROB
    if do_aug and cfg.TRAIN.DISTORT.USE_GRAYSCALE and to_go:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    # fit to target shape if needed
    if im.shape[0] != target_shape[1] or im.shape[1] != target_shape[0]:
        im = cv2.resize(im, target_shape, interpolation=cv2.INTER_AREA)

    im = convert_color_space(im, cfg, inverse=False)
    return im