import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
#
__C.DATASET = edict()
__C.DATASET.DATAROOT = ''
__C.DATASET.VIDEO_FORMAT = 'avi'
__C.DATASET.TEST_SPLIT_NAME = 'video_list_test.txt'
__C.DATASET.CLIP_LEN = 8
__C.DATASET.CLIP_STRIDE = 4

# data pre-processing options
#
__C.DATA_TRANSFORM = edict()
#__C.DATA_TRANSFORM.RESIZE_OR_CROP = 'resize_and_crop'
__C.DATA_TRANSFORM.LOADSIZE = 256 #178
__C.DATA_TRANSFORM.FINESIZE = 224 #112
#__C.DATA_TRANSFORM.FLIP = True
#__C.DATA_TRANSFORM.NORMALIZE_MEAN = (0.485, 0.456, 0.406)
#__C.DATA_TRANSFORM.NORMALIZE_STD = (0.229, 0.224, 0.225)

# model
__C.MODEL = edict()
__C.MODEL.PRETRAINED_ENCODER_WEIGHTS = "./experiments/ckpt/rgb_charades.pt"

# Testing options
#
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 1

# MISC
__C.WEIGHTS = ''
__C.RESUME = ''
__C.EXP_NAME = 'exp'
__C.SAVE_DIR = ''
__C.NUM_WORKERS = 8
__C.THREADS_PER_GPU = 3
__C.NUM_GPUS = 4

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k in a:
        # a must specify keys that are in b
        v = a[k]
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
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
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
