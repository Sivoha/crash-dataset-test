# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import random
import shutil
import operator
from PIL import Image
import platform
import json
import base64
import typing as t
import zlib
import re
import tqdm


random.seed(2016)
np.random.seed(2016)

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
FEATURES_PATH = ROOT_PATH + 'features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3), protocol=4)


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_rgb(im, name='image'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('acc', 'val_acc')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def read_single_image(path):
    img = cv2.imread(path)
    return img


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model' or layer_type == 'Functional':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def dice_metric_score(real, pred):
    assert real.shape == pred.shape
    r = real.astype(np.bool)
    p = (pred > 0.5).astype(np.bool)
    r_sum = r.sum()
    p_sum = p.sum()
    if r_sum == 0 and p_sum == 0:
        return 1.0
    if r_sum == 0:
        return 0.0
    if p_sum == 0:
        return 0.0

    intersection = np.logical_and(r, p).sum()
    return 2 * intersection / (r_sum + p_sum)


def normalize_array(cube, new_max, new_min):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(cube), np.max(cube)
    if maximum - minimum != 0:
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        cube = m * cube + b
    return cube


def reduce_model(model_path):
    from kito import reduce_keras_model
    from keras.models import load_model

    m = load_model(model_path)
    m_red = reduce_keras_model(m)
    m_red.save(model_path[:-3] + '_reduced.h5')


def get_image_size(path):
    im = Image.open(path)
    width, height = im.size
    return height, width, im.mode


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s


def jackard_metric_score(real, pred):
    assert real.shape == pred.shape
    r = real.astype(np.bool)
    p = (pred > 0.5).astype(np.bool)
    r_sum = r.sum()
    p_sum = p.sum()
    if r_sum == 0 and p_sum == 0:
        return 1.0
    if r_sum == 0:
        return 0.0
    if p_sum == 0:
        return 0.0

    intersection = np.logical_and(r, p).sum()
    return intersection / (r_sum + p_sum - intersection)


def read_video(f, verbose=False, output_info=False):
    cap = cv2.VideoCapture(f)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    frame_list = []
    if verbose:
        print('ID: {} Video length: {} Width: {} Height: {} FPS: {}'.format(os.path.basename(f), length, width, height, fps))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame.copy())
        current_frame += 1

    frame_list = np.array(frame_list, dtype=np.uint8)
    cap.release()
    if output_info:
        return frame_list, fps

    return frame_list