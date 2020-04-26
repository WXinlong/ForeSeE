from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from lib.utils.collections import AttrDict
from lib.utils.misc import get_run_name
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions

# ---------------------------------------------------------------------------- #
# Load parse for training, val, and test
# ---------------------------------------------------------------------------- #
train_opt = TrainOptions()
train_args = train_opt.parse()
train_opt.print_options(train_args)

val_opt = ValOptions()
val_args = val_opt.parse()
val_args.batchsize = 1
val_args.thread = 0
val_opt.print_options(val_args)

__C = AttrDict()
# Consumers can get config by:
cfg = __C

# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead
# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
__C.EXP_NAME = os.path.dirname(__file__).split('/')[-1]
__C.DATASET = train_args.dataset
# "Fun" fact: the history of where these values comes from is lost (From Detectron lol)
__C.RGB_PIXEL_MEANS = (102.9801, 115.9465, 122.7717)
__C.RGB_PIXEL_VARS = (1, 1, 1)
__C.CROP_SIZE = (385, 513) if 'kitti' in train_args.dataset else (385, 385)  #height * width

# ---------------------------------------------------------------------------- #
# Models configurations
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
__C.MODEL.INIT_TYPE = 'xavier'
# Configure the model type for the encoder, e.g.ResNeXt50_32x4d_body_stride16
__C.MODEL.ENCODER = train_args.encoder
__C.MODEL.MODEL_REPOSITORY = 'pretrained_model'
__C.MODEL.PRETRAINED_WEIGHTS = train_args.pretrained_model
__C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = True

# Configure resnet and resnext
__C.MODEL.RESNET_BOTTLENECK_DIM = [64, 256, 512, 1024, 2048] if 'ResNeXt' in train_args.encoder else [32, 24, 32, 96, 320]
__C.MODEL.RESNET_BLOCK_DIM = [64, 64, 128, 256, 512]
# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
__C.MODEL.RESNET_STRIDE_1X1 = True
# Set bn type of resnet, bn->batch normalization, affine->affine transformation
__C.MODEL.RESNET_BN_TYPE = 'bn'

# Freeze the batch normalization layer of pretrained model
__C.MODEL.FREEZE_BACKBONE_BN = False
# Configure the decoder
__C.MODEL.FCN_DIM_IN = [512, 256, 256, 256, 256, 256] if 'ResNeXt' in train_args.encoder else [128, 64, 64, 64, 64, 64]
__C.MODEL.FCN_DIM_OUT = [256, 256, 256, 256, 256] if 'ResNeXt' in train_args.encoder else [64, 64, 64, 64, 64]
__C.MODEL.LATERAL_OUT = [512, 256, 256, 256] if 'ResNeXt' in train_args.encoder else [128, 64, 64, 64]


# Configure input and output channel of the model
__C.MODEL.ENCODRE_INPUT_C = 3
__C.MODEL.DECODER_OUTPUT_C = train_args.decoder_out_c

# Configure weight for different losses
__C.MODEL.DIFF_LOSS_WEIGHT = 6

# ---------------------------------------------------------------------------- #
# Data configurations
# ---------------------------------------------------------------------------- #
__C.DATA = AttrDict()
__C.DATA.DATA_SET = train_args.dataset
# Minimum depth
__C.DATA.DATA_MIN = 0.01 if 'nyu' in train_args.dataset else 0.015
# Maximum depth
__C.DATA.DATA_MAX = 1.7 if 'nyu' in train_args.dataset else 1.0
# Minimum depth in log space
__C.DATA.DATA_MIN_LOG = np.log10(__C.DATA.DATA_MIN)
# Interval of each range
__C.DATA.DEPTH_RANGE_INTERVAL = (np.log10(__C.DATA.DATA_MAX) - np.log10(__C.DATA.DATA_MIN)) / __C.MODEL.DECODER_OUTPUT_C
# Depth class
__C.DATA.DEPTH_CLASSES = np.array([__C.DATA.DATA_MIN_LOG + __C.DATA.DEPTH_RANGE_INTERVAL * (i + 0.5) for i in range(__C.MODEL.DECODER_OUTPUT_C)])
__C.DATA.WCE_LOSS_WEIGHT = [[np.exp(-0.2 * (i - j) ** 2) for i in range(__C.MODEL.DECODER_OUTPUT_C)]
                            for j in np.arange(__C.MODEL.DECODER_OUTPUT_C)]
__C.DATA.LOAD_MODEL_NAME = train_args.load_ckpt
# ---------------------------------------------------------------------------- #
# Training configurations
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
# Load run name, which is the combination of running time and host name
__C.TRAIN.RUN_NAME = get_run_name()
__C.TRAIN.OUTPUT_ROOT_DIR = './outputs'
#__C.TRAIN.OUTPUT_ROOT_DIR = '/mnt/cephfs_hl/vc/wxl/depth/' + __C.EXP_NAME
# Dir for checkpoint and logs
__C.TRAIN.LOG_DIR = os.path.join(__C.TRAIN.OUTPUT_ROOT_DIR, train_args.dataset + '_' + cfg.TRAIN.RUN_NAME)
# Differ the learning rate between encoder and decoder
__C.TRAIN.DIFF_LR = train_args.scale_decoder_lr
__C.TRAIN.BASE_LR = train_args.lr
__C.TRAIN.MAX_ITER = 0
# Set training epoches, end at the last epoch of list
__C.TRAIN.EPOCH = train_args.epoch
# Snapshot (model checkpoint) period
__C.TRAIN.SNAPSHOT_ITERS = 6000
__C.TRAIN.VAL_STEP = 6000
__C.TRAIN.BATCH_SIZE = train_args.batchsize
__C.TRAIN.GPU_NUM = 1
# Steps for LOG interval
__C.TRAIN.LOG_INTERVAL = 20
__C.TRAIN.LR_DECAY_MILESTONES = __C.TRAIN.EPOCH[1:-1]



def merge_cfg_from_file(datasize, gpu_num):
    __C.TRAIN.MAX_ITER = round(datasize / __C.TRAIN.BATCH_SIZE + 0.5) * __C.TRAIN.EPOCH[-1]
    __C.TRAIN.GPU_NUM = gpu_num







