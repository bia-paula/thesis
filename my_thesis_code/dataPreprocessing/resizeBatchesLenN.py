import os.path

import laplacian_foveation as fv
import numpy as np
import math
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import Sequence
from time import time
import pickle

from glob import glob

import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config

def init_batch(size):
    d = {"rnn_x": np.zeros((0, size + 1, config.fmap_size[0], config.fmap_size[1], config.fmap_size[2])),
         "label_encodings": np.zeros((0, len(config.classes))),
         "rnn_y": np.zeros((0, size + 1, config.fmap_size[0] * config.fmap_size[1]))}
    return d


def append_to_batch(d, data, start, finish):
    rnn_x = data["rnn_x"][start:finish]
    label_encodings = data["label_encodings"][start:finish]
    rnn_y = data["rnn_y"][start:finish]

    d["rnn_x"] = np.append(d["rnn_x"], rnn_x, axis=0)
    d["label_encodings"] = np.append(d["label_encodings"], label_encodings, axis=0)
    d["rnn_y"] = np.append(d["rnn_y"], rnn_y, axis=0)

    return d


def copy_dicts(from_d, size):
    to_d = init_batch(size)
    to_d["rnn_x"] = np.append(to_d["rnn_x"], from_d["rnn_x"], axis=0)
    to_d["label_encodings"] = np.append(to_d["label_encodings"], from_d["label_encodings"], axis=0)
    to_d["rnn_y"] = np.append(to_d["rnn_y"], from_d["rnn_y"], axis=0)
    return to_d


if __name__ == "__main__":

    batchSize = 256
    fovea_size = 75
    files_dir = "/Volumes/DropSave/Tese/dataset/sequences_by_nfixations"
    save_dir = "/Volumes/DropSave/Tese/dataset/sequences_by_nfixations/batchSize" + str(batchSize)


    for trainval in ["train", "valid"]:

        print(trainval)

        files_name = trainval + "_scanpaths_fov" + str(fovea_size) + "_filtered_length"
        for size in range(7):

            names = []
            count = 0

            print("***** Size:", size, "*****")

            files_size = files_name + str(size)
            files_path = glob(os.path.join(files_dir, files_size + "*"))

            this_batch = init_batch(size)
            next_batch = init_batch(size)

            this_len = 0
            next_len = 0
            print(files_path)
            for file in files_path:

                this_batch = copy_dicts(next_batch, size)
                next_batch = init_batch(size)
                this_len = next_len
                next_len = 0

                data = np.load(file, allow_pickle=True)
                print("New file:", data["rnn_x"].shape[0])
                remaining_in_file = data["rnn_x"].shape[0]
                file_size = remaining_in_file

                if remaining_in_file == 0:
                    next_batch = copy_dicts(this_batch, size)
                    next_len = this_len

                while remaining_in_file > 0:
                    to_append = min(batchSize - this_len, remaining_in_file)
                    start = file_size-remaining_in_file
                    this_batch = append_to_batch(this_batch, data, start, start + to_append)
                    remaining_in_file -= to_append
                    this_len += to_append

                    if this_len == batchSize:
                        file_name =  "_".join(file.split("/")[-1].split(".")[0].split("_")[:-1]) + "." + str(count)
                        np.savez_compressed(os.path.join(save_dir, file_name), rnn_x=this_batch["rnn_x"],
                                            label_encodings=this_batch["label_encodings"], rnn_y=this_batch["rnn_y"])
                        print("\tSaved:", this_batch["rnn_x"].shape[0])
                        names.append(file_name)
                        count += 1
                        this_batch = init_batch(size)
                        this_len = 0
                        next_batch = init_batch(size)
                        next_len = 0
                    else:
                        next_batch = copy_dicts(this_batch, size)
                        next_len = this_len
                        print("\tCarry:", next_len)

            if next_len > 0:
                file_name = "_".join(file.split("/")[-1].split(".")[0].split("_")[:-1]) + "." + str(count)
                np.savez_compressed(os.path.join(save_dir, file_name), rnn_x=next_batch["rnn_x"],
                                    label_encodings=next_batch["label_encodings"], rnn_y=next_batch["rnn_y"])
                print("\tSaved smaller:", next_batch["rnn_x"].shape)
                print(file_name)
                names.append(file_name)
                count += 1

            count = len(names)
            a = range(5)
            b = range(math.ceil(float(count)/float(5)))
            c = [(x, y) for x in a for y in b]
            print(names)
            for i, name in enumerate(names):
                old_dir = os.path.join(save_dir, name + ".npz")
                new_name = name.split(".")[0] + "." + str(c[i][0]) + "." + str(c[i][1]) + ".npz"
                new_dir = os.path.join(save_dir, new_name)
                os.rename(old_dir, new_dir)
                "/Volumes/DropSave/Tese/dataset/sequences_by_nfixations/batchSize256/train_scanpaths_fov100_filtered_length0.0.npz"

