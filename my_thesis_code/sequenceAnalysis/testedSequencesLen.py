import json
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np
from sequenceDistances import get_distances
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from glob import glob
import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')

from dataPreprocessing.foveateImages import ind2gridcoord, gridcoord2realcoord
from dataPreprocessing.foveateImages import smooth_foveate, one_hot_encoder, realcoord2gridcoord, gridcoord2realcoord, \
    ind2gridcoord, fixated_object
from fixationPrediction.predict import beam_search
from evaluation.accuracyBeatrizC import bbox2cellsbboxcoord
import config

if __name__ == "__main__":

    '''best_path = "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label/testing"
    worst_path = "/Volumes/DropSave/Tese/trainedModels/fov75_batch64_onehot_rnny_heatmap_label2d/testing"
    dists = np.empty([6, 0])
    unique = []

    for images_best in glob(os.path.join(worst_path, "*", "*")):
        data = np.load(images_best)["seqs"][0]
        real_X = np.zeros((7,))
        real_Y = np.zeros((7,))

        for i, ind in enumerate(data):

            grid = ind2gridcoord(ind)
            real_X[i], real_Y[i] = gridcoord2realcoord(grid[0], grid[1])

        unique.append(len(set(data)))
        dists = np.append(dists, get_distances(real_X, real_Y, 1), axis=1)

    x = range(1, 7)

    hist = np.nanmean(dists, axis=1)
    plt.bar(x, hist)
    plt.show()

    hist = np.nanmedian(dists, axis=1)
    plt.bar(x, hist)
    plt.show()

    plt.hist(unique, bins=range(9))
    plt.show()'''

    for size in range(7):

        paths = glob(
            "/Volumes/DropSave/Tese/dataset/sequences_by_nfixations/train_scanpaths_fov100_filtered_length" + str(
                size) + "*")

        batch = 128

        filename = "train_scanpaths_fov100_batch" + str(batch) + ".length" + str(size) + "."  # <---------------

        # ***********************************
        paths.sort()

        #print(paths)

        X = np.zeros((0, size+1, 10, 16, 512))
        Task = np.zeros((0, 18))
        Y = np.zeros((0, size+1, 160))

        #print(X.shape)

        k = 0
        lens = []

        for file in paths:
            with np.load(file) as data:
                # new_X = data['rnn_x']
                new_Task = data['label_encodings']
                # new_Y = data['rnn_y']
                k += 1
                lens.append(new_Task.shape[0])

        print("Size", size)
        print(lens)


