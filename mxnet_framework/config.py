import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Training options
#
__C.TRAIN = edict()
__C.TRAIN.DATASETS = []

__C.TRAIN.DISTORT = edict()

__C.TRAIN.DISTORT.USE_PAD = False
__C.TRAIN.DISTORT.PAD_SIZE = 0

__C.TRAIN.DISTORT.USE_ROTATE = False
__C.TRAIN.DISTORT.ROTATE_PROB = 0.5
__C.TRAIN.DISTORT.MAX_ROTATE = 10

__C.TRAIN.DISTORT.USE_DEFORM = False
__C.TRAIN.DISTORT.DEFORM_PROB = 0.5
__C.TRAIN.DISTORT.MAX_DEFORM = 1.1

__C.TRAIN.DISTORT.USE_FLIP = False

__C.TRAIN.DISTORT.USE_CROP = True
__C.TRAIN.DISTORT.CROP_REGION = 1.0

__C.TRAIN.DISTORT.USE_SCALE = True
__C.TRAIN.DISTORT.SCALE_PROB = 0.5
__C.TRAIN.DISTORT.MAX_SCALE = 0.05
__C.TRAIN.DISTORT.SCALE_MIN = 0.95
__C.TRAIN.DISTORT.SCALE_MAX = 1.05

__C.TRAIN.DISTORT.USE_GRAYSCALE = False
__C.TRAIN.DISTORT.GRAYSCALE_PROB = 0.1

__C.TRAIN.DISTORT.USE_BRIGHTNESS = False
__C.TRAIN.DISTORT.BRIGHTNESS_PROB = 0.25
__C.TRAIN.DISTORT.BRIGHT_MIN_ALPHA = 0.95
__C.TRAIN.DISTORT.BRIGHT_MAX_ALPHA = 1.05

__C.TRAIN.DISTORT.BRIGHT_MIN_BETA = 0.
__C.TRAIN.DISTORT.BRIGHT_MAX_BETA = 0.

__C.TRAIN.DISTORT.USE_WHITE_BALANCE = False
__C.TRAIN.DISTORT.WHITE_BALANCE_PROB = 0.5
__C.TRAIN.DISTORT.WHITE_BALANCE_COEFF = 0.01

__C.TRAIN.DISTORT.USE_INVERT = False
__C.TRAIN.DISTORT.INVERT_PROB = 0.5

__C.TRAIN.DISTORT.USE_CHANNEL_SHUFFLE = False
__C.TRAIN.DISTORT.CHANNEL_SHUFFLE_PROB = 0.5

__C.TRAIN.DISTORT.USE_JPEG = False
__C.TRAIN.DISTORT.JPEG_PROB = 0.25
__C.TRAIN.DISTORT.JPEG_MIN_QUALITY = 12
__C.TRAIN.DISTORT.JPEG_MAX_QUALITY = 20

__C.TRAIN.DISTORT.USE_NOISE = False
__C.TRAIN.DISTORT.NOISE_PROB = 0.5
__C.TRAIN.DISTORT.NOISE_SIGMA = 0.1

__C.TRAIN.DISTORT.USE_LOW_RES = False
__C.TRAIN.DISTORT.LOW_RES_PROB = 0.25
__C.TRAIN.DISTORT.LOW_RES_FACTOR_MIN = 1.5
__C.TRAIN.DISTORT.LOW_RES_FACTOR_MAX = 2.5

__C.TRAIN.DEBUG_IMAGES = ''
__C.TRAIN.LOADER_WORKERS = 0
__C.TRAIN.CSV_PARSER_WORKERS = 0

__C.TRAIN.EPOCHS = 128
__C.TRAIN.WEIGHT_DECAY = 0.0001
__C.TRAIN.BASE_LR = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.OPTIMIZER = 'sgd'
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.EPOCHS_STEPS = [128]
__C.TRAIN.FACTOR_D = 0.1

__C.TRAIN.PRETRAINED = ''
__C.TRAIN.PRETRAINED_EPOCH = 0

__C.TRAIN.SNAPSHOT_PERIOD = 10
__C.TRAIN.DISPLAY_ITERS = 20

#
# Testing options
#
__C.TEST = edict()
__C.TEST.DATASETS = []
__C.TEST.BATCH_SIZE = 128
__C.TEST.DEBUG_IMAGES = ''
__C.TEST.LOADER_WORKERS = 0
__C.TEST.CSV_PARSER_WORKERS = 0

#
# Validation options
#
__C.VALIDATION = edict()
__C.VALIDATION.DATASETS = []
__C.VALIDATION.REPORT_INTERMEDIATE = False
__C.VALIDATION.INTERVAL = 1000
__C.VALIDATION.DEBUG_IMAGES = ''
__C.VALIDATION.LOADER_WORKERS = 0
__C.VALIDATION.CSV_PARSER_WORKERS = 0
__C.VALIDATION.BATCH_SIZE = 128

#
# CSV Input format
#
__C.INPUT_FORMAT = edict()
__C.INPUT_FORMAT.FORMAT = 'labels'

#
# Common
#
__C.PIXEL_MEANS = [104.00699, 116.66877, 122.67892]
__C.MEAN_IMG = np.array([])
__C.MEAN_STD = np.array([])
__C.IMAGE_MEANS_PATH = ''
__C.IMAGE_STD_PATH = ''
__C.MEANS_TYPE = 'pixel' # or 'image'
__C.SUBTR_MEANS = False
__C.GRAYSCALE = False
__C.COLOR_SPACE = 'rgb' # can be bgr
__C.PLOT_GRAPH = True
__C.CACHE_MAX_SIZE = 0 # in GB

__C.MAX_LABELS = 6
__C.MIN_LABELS = 4
__C.CLS_NUM = 36

__C.MAX_DEBUG_IMAGES = 200

# For reproducibility
__C.RNG_SEED = 3

# Default GPU device id
__C.GPU_IDS = []
__C.GPU_NUM = 4

# Size
__C.PREPARE_SCALE = None
__C.PREPARE_SIZE = None
__C.INPUT_SHAPE = [330, 150]
__C.NORM_COEFF = 1.0

def get_output_dir(suffix, net):
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'exps', __C.EXP_DIR, 'output', suffix))
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    if type(a) is not edict:
        return

    for k, v in a.items():
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k])
        is_none = b[k] is None or v is None
        if old_type is not type(v) and not is_none:
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k))

        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value