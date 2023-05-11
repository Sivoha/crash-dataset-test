# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *

KFOLD_SPLIT_FILE = OUTPUT_PATH + 'kfold_split_5_42_upd.csv'
RUN_NUMBER = 1

EPOCHS = 30
if RUN_NUMBER == 1:
    SHAPE_SIZE = (3, 384, 384)
    DROPOUT = 0.5
    USE_HARD_AUG = True
    OPTIMIZER = 'AdamW'
elif RUN_NUMBER == 2:
    SHAPE_SIZE = (3, 360, 640)
    DROPOUT = 0.3
    USE_HARD_AUG = True
    OPTIMIZER = 'AdamW'


NUM_CLASSES = 1
if SHAPE_SIZE[-1] <= 224:
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALID = 64
elif SHAPE_SIZE[-1] <= 384:
    BATCH_SIZE_TRAIN = 48
    BATCH_SIZE_VALID = 48
elif SHAPE_SIZE[-1] <= 512:
    BATCH_SIZE_TRAIN = 12
    BATCH_SIZE_VALID = 12
elif SHAPE_SIZE[-1] <= 768:
    BATCH_SIZE_TRAIN = 10
    BATCH_SIZE_VALID = 10
elif SHAPE_SIZE[-1] <= 1024:
    BATCH_SIZE_TRAIN = 7
    BATCH_SIZE_VALID = 7
elif SHAPE_SIZE[-1] <= 1280:
    BATCH_SIZE_TRAIN = 5
    BATCH_SIZE_VALID = 5


STEPS_PER_EPOCH = 1000
EARLY_STOPPING = 45
USE_ONE_CYCLE_SCHEDULER = True
THREADS = 1
if RUN_NUMBER == 1:
    START_LEARNING_RATE = 0.5
elif RUN_NUMBER == 2:
    START_LEARNING_RATE = 1
elif RUN_NUMBER == 3:
    START_LEARNING_RATE = 1
elif RUN_NUMBER == 4:
    START_LEARNING_RATE = 1
elif RUN_NUMBER == 5:
    START_LEARNING_RATE = 1
elif RUN_NUMBER == 6:
    START_LEARNING_RATE = 1