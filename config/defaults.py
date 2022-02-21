from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.PRINT_FREQ = 100
_C.SEED = 0

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'deeplabv3+'
_C.MODEL.BACKBONE = 'mobilenet_v2'
_C.MODEL.PRETRAINED_WEIGHTS = ''

# common params for CONTROLLER
_C.CONTROLLER = CN()
_C.CONTROLLER.NAME = 'controller'
_C.CONTROLLER.LOSS = 'ppo'
_C.CONTROLLER.PENALTY = 0.00001
_C.CONTROLLER.L = 2
_C.CONTROLLER.M = 6
_C.CONTROLLER.T = 2
_C.CONTROLLER.C = 2.5
_C.CONTROLLER.NUM_MAGS = 10
_C.CONTROLLER.EXCLUDE_OPS_NUM = 0
_C.CONTROLLER.EXCLUDE_OPS = []

# common params for DISCRIMINATOR
_C.DISCRIMINATOR = CN()
_C.DISCRIMINATOR.NAME = 'momentum_feature'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = './dataset'
_C.DATASET.NAME = 'cifar10'
_C.DATASET.TRAINSET = ''
_C.DATASET.TESTSET = ''

# Domain Generalization related params
_C.DATASET.DG = CN()
_C.DATASET.DG.TRAIN = [1, 2, 3]
_C.DATASET.DG.TEST = [4]

# train
_C.TRAIN = CN()

_C.TRAIN.LR = 0.1
_C.TRAIN.WD = 0.0004
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.WARMUP_EPOCH = 0
_C.TRAIN.END_EPOCH = 200

_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE = _C.TRAIN.BATCH_SIZE
_C.TEST.MODEL_DIR = ''


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)