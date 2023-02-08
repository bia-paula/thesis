import laplacian_foveation as fv
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import Sequence
from time import time

from glob import glob

import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config


def is_second_half(rnn_y):
    return not ((rnn_y[6] == rnn_y[5]).all() and (rnn_y[5] == rnn_y[4]).all() and (rnn_y[4] == rnn_y[3]).all())


def add_to_half(d, rnn_x, label_enc, rnn_y):
    d["rnn_x"] = np.append(d["rnn_x"], np.expand_dims(rnn_x, axis=0), axis=0)
    d["label_encodings"] = np.append(d["label_encodings"], np.expand_dims(label_enc, axis=0), axis=0)
    d["rnn_y"] = np.append(d["rnn_y"], np.expand_dims(rnn_y, axis=0), axis=0)
    return d


if __name__ == "__main__":

    for trainval in ["valid"]:

        dirs = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/"+trainval+"_scanpaths_fov100*"
        d_vals = range(7)
        done_paths=[]
        for i in d_vals:
            done = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/valid_scanpaths_fov100_batch256.0."+str(i)
            done_paths += glob(done+"*")
        done = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/train_scanpaths_fov100_batch256.3.[0-8]"
        done_paths += glob(done + "*")
        paths_before = glob(dirs)
        paths = [p for p in paths_before if p not in done_paths]
        paths.sort()
        for file_i, path in enumerate(paths):
            print(path)
            data = np.load(path)
            print(data["rnn_x"].shape)

            size_x = list(data["rnn_x"].shape)
            size_x[0] = 0
            size_enc = list(data["label_encodings"].shape)
            size_enc[0] = 0
            size_y = list(data["rnn_y"].shape)
            size_y[0] = 0

            first = {"rnn_x": np.zeros(tuple(size_x)), "label_encodings": np.zeros(tuple(size_enc)),
                     "rnn_y": np.zeros(tuple(size_y))}
            second = {"rnn_x": np.zeros(tuple(size_x)), "label_encodings": np.zeros(tuple(size_enc)),
                      "rnn_y": np.zeros(tuple(size_y))}

            for i in range(data["label_encodings"].shape[0]):
                if is_second_half(data["rnn_y"][i]):
                    add_to_half(second, data["rnn_x"][i], data["label_encodings"][i], data["rnn_y"][i])
                else:
                    add_to_half(first, data["rnn_x"][i], data["label_encodings"][i], data["rnn_y"][i])

            print(first["rnn_x"].shape)
            print(second["rnn_x"].shape)
            file_name = path.split("/")[-1]
            save_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/half_sequence/"
            save_file_first = save_dir + "first/before/" + file_name
            save_file_second = save_dir + "second/before/" + file_name

            np.savez(save_file_first, rnn_x=first["rnn_x"], label_encodings=first["label_encodings"],
                     rnn_y=first["rnn_y"])
            np.savez(save_file_second, rnn_x=second["rnn_x"], label_encodings=second["label_encodings"],
                     rnn_y=second["rnn_y"])

            print("File", file_i, "/", len(paths))


