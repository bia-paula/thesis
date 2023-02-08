import laplacian_foveation as fv
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import math
import time
from datetime import timedelta
from glob import glob
from math import pi, exp
import scipy
import os

import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import ind2gridcoord, gridcoord2ind

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

def gaussianfunc(x, y, miu_x, miu_y, sigma):
    d = ((x - miu_x) ** 2) + ((y - miu_y) ** 2)
    f = (1 / (2 * pi * sigma)) * exp(-d / (2 * (sigma ** 2)))
    return f

def gaussian2d(miu_x, miu_y, sigma):
    matrice = np.zeros((config.fmap_size[0] * config.fmap_size[1]))  #(10, 16) (y, x)
    for x in range(config.fmap_size[1]):
        for y in range(config.fmap_size[0]):
            d = ((x - miu_x) ** 2) + ((y - miu_y) ** 2)
            matrice[gridcoord2ind(x, y)] = (1 / (2 * pi * sigma)) * exp(-d / (2 * (sigma ** 2)))
    return matrice

def onehotrnny2gaussian2d(rnn_y, sigma):
    x, y = ind2gridcoord(np.argmax(rnn_y))
    pos = [y, x]
    pos = np.expand_dims(np.expand_dims(np.array(pos), axis=1).repeat(10, axis=1), axis=2).repeat(16, axis=2)
    m = (scipy.mgrid[0: 10, 0: 16] - pos) ** 2
    sum = m[0] + m[1]
    e = np.exp(- sum / (2 * sigma ** 2))
    gauss = np.array(e / (2 * sigma * np.pi))

    return gauss.flatten()

def onehotrnny2distdamping(rnn_y):
    x, y = ind2gridcoord(np.argmax(rnn_y))
    pos = [y, x]
    pos = np.expand_dims(np.expand_dims(np.array(pos), axis=1).repeat(10, axis=1), axis=2).repeat(16, axis=2)
    m = (scipy.mgrid[0: 10, 0: 16] - pos) ** 2
    sum = np.array(m[0] + m[1], dtype="float64")
    rnn_y = np.reciprocal(1.0 + sum)
    return rnn_y.flatten()






def getFPgridcoord(rnn_y):
    ind = np.argmax(rnn_y)
    x, y = ind2gridcoord(ind)
    return x, y


if __name__ == '__main__':

    first_time = time.time()
    local_time = time.ctime(first_time)
    print("Local time:", local_time)
    # Training directories
    batch_size = 32  # <-----------
    train_path = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + str(batch_size)
    save_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + str(batch_size) + \
               "/normalFPGT/"

    files = glob(train_path + "/*.npz")

    sigma = 1# <-------- try different values
    count = 0

    for file in files:
        # Load file
        with open(file) as fp:
            rnn_y_onehot = np.load(file)["rnn_y"]

        exps_shape = rnn_y_onehot.shape[0]
        rnn_y_gaussian = np.zeros((exps_shape, 7, 160))
        inds_gaussian = np.zeros((exps_shape, 7))
        coord_gaussian = np.zeros((exps_shape, 7, 2))

        for obs in range(rnn_y_onehot.shape[0]):
            for fp_ind in range(rnn_y_onehot.shape[1]):
                miu_x, miu_y = getFPgridcoord(rnn_y_onehot[obs, fp_ind])
                rnn_y_gaussian[obs, fp_ind] = gaussian2d(miu_x, miu_y, sigma)

        name = file.split("/")[-1]
        save_file = save_dir + name
        np.savez_compressed(save_file, rnn_y_normal=rnn_y_gaussian)
        count += 1
        print(count, "/", len(files))

    last_time = time.time()
    print("TOTAL TIME: ")
    print(timedelta(seconds=last_time - first_time))
    local_time = time.ctime(last_time)


