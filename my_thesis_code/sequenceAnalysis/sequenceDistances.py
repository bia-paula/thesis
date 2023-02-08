import json
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import smooth_foveate, one_hot_encoder, realcoord2gridcoord, gridcoord2realcoord, \
    ind2gridcoord, fixated_object
from fixationPrediction.predict import beam_search
from evaluation.accuracyBeatrizC import bbox2cellsbboxcoord
import config


def get_distances(X, Y, trainval):
    dist = np.full([6, 1], np.nan)

    for i in range(1, min(len(X), 7)):
        dist[i - 1] = np.sqrt((X[i] - X[i - 1]) ** 2 + (Y[i] - Y[i - 1]) ** 2)
    if not trainval:
        dist = dist * 512/1680
    return dist


if __name__ == '__main__':

    data_dir = "/Volumes/DropSave/Tese/dataset"
    name_pre = "human_scanpaths_TP_"

    file_seq = []
    files_lens = []
    file_seq_zeros = []

    for f_idx, f_name in enumerate(["test.json", "trainval_train.json", "trainval_valid.json"]):

        seq = np.empty([6, 0])
        lens = []

        with open(os.path.join(data_dir, name_pre + f_name)) as fp:
            exps = json.load(fp)

        # Iterate over experiments
        for obs in exps:
            flag = 0
            for i in range(min(obs["length"], 7)):
                if fixated_object(obs["X"][i], obs["Y"][i], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                                  obs["bbox"][3]):
                    flag = 1
                    break

            # Consider only obs fixate in 6
            if flag:
                dist = get_distances(obs["X"], obs["Y"], f_idx)
                seq = np.append(seq, dist, axis=1)
                lens.append(obs["length"])

        file_seq.append(seq)
        file_seq_zeros.append(np.nan_to_num(seq))
        files_lens.append(np.array(lens))

    x = range(1, 7)
    '''
    for i in range(1):
        hist = np.nanmean(file_seq[i], axis=1)
        plt.bar(x, hist)
        plt.show()
        hist = np.nanmean(file_seq_zeros[i], axis=1)
        plt.bar(x, hist)
        plt.show()
        hist = np.nanmedian(file_seq[i], axis=1)
        plt.bar(x, hist)
        plt.show()
        hist = np.nanmedian(file_seq_zeros[i], axis=1)
        plt.bar(x, hist)
        plt.show()'''


    plt.hist(files_lens[0], bins=range(9))
    plt.show()
    '''plt.hist(files_lens[1], bins=range(9))
    plt.show()
    plt.hist(files_lens[2], bins=range(9))
    plt.show()

    print(np.sqrt(512**2 + 320**2))'''
