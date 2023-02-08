import os.path

import laplacian_foveation as fv
import numpy as np
from PIL import Image, ImageDraw

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import math
import time
from datetime import timedelta
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config

from dataPreprocessing.foveateImages import fixated_object, realcoord2gridcoord, gridcoord2ind, one_hot_encoder

if __name__ == "__main__":

    first_time = time.time()
    local_time = time.ctime(first_time)
    print("Local time:", local_time)

    trainval_dir = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_"
    dcb_dir = "/Users/beatrizpaula/Desktop/images_HL_DCBs/DCBs"
    save_path = "/Users/beatrizpaula/Desktop/images_HL_DCBs/hardPanoptic/batchSize256"

    for trainval in ["train", "valid"]:

        file_path = trainval_dir + trainval + ".json"
        with open(file_path) as fp:
            data = json.load(fp)

        count_file = 0
        obs_in_file = 0
        count_out = 0

        rnn_x = []
        rnn_y = []
        label_encodings = []
        initf_time = time.time()

        for count_total, obs in enumerate(data):
            obs_start = time.time()
            flag = 0
            # Check how many fp it took to detect object
            for i in range(min(obs["length"], 7)):
                if fixated_object(obs["X"][i], obs["Y"][i], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                                  obs["bbox"][3]):
                    flag=1
                    break

            if not flag:
                count_out += 1
                print(count_out)
                continue

            class_name = obs["task"]
            image_id = obs["name"].split(".")[0]

            # holder for fp indexes ground truth
            gt_fixs = np.zeros((0, config.fmap_size[0] * config.fmap_size[1]))

            obs_dcbs = dict()

            for hl in ["H", "L2", "L5", "L7"]:
                fp = os.path.join(dcb_dir, class_name, image_id + "_" + hl + '.npz')
                with np.load(fp) as value:
                    obs_dcbs[hl] = value["dcb"]

            for i_fp in range(1, 7 + 1):
                true_i_fp = i_fp
                # For shorter sequences save last fp of sequence until seq len is completed
                if i_fp > obs["length"] - 1:
                    true_i_fp = obs["length"] - 1

                # Last fp prediction points to same point
                elif i_fp == config.sequence_len:
                    true_i_fp = config.sequence_len - 1
                # Get coord index of next fp
                next_fix = np.zeros(config.fmap_size[0] * config.fmap_size[1])

                # next_i = i_fp - 1
                next_x = obs["X"][true_i_fp]  # [next_i]
                next_y = obs["Y"][true_i_fp]  # [next_i]
                grid_next_x, grid_next_y = realcoord2gridcoord(next_x, next_y)
                idx_next = gridcoord2ind(grid_next_x, grid_next_y)
                # one hot encode next fixation
                next_fix[idx_next] = 1

                next_fix = np.expand_dims(next_fix, axis=0)
                gt_fixs = np.append(gt_fixs, next_fix, axis=0)


            rnn_x.append(obs_dcbs)
            label_encodings.append(one_hot_encoder(class_name))
            rnn_y.append(gt_fixs)



            obs_in_file += 1

            obs_end = time.time()
            print(str(count_total) + "/" + str(len(data)), " -> (", str(obs["length"]), ") ",
                  timedelta(seconds=obs_end - obs_start))

            if obs_in_file == 256:
                rnn_x = np.array(rnn_x)
                label_encodings = np.array(label_encodings)
                rnn_y = np.array(rnn_y)

                file_name = trainval + '_scanpaths_batch256.' + str(count_file)

                np.savez_compressed(os.path.join(save_path, file_name), rnn_x=rnn_x,
                                    label_encodings=label_encodings, rnn_y=rnn_y)
                obs_in_file = 0
                count_file += 1

                rnn_x = []
                label_encodings = []
                rnn_y = []

                print("***** SAVED FILES " + str(count_file) + " *****")
                endf_time = time.time()
                print(timedelta(seconds=endf_time - initf_time))
                initf_time = time.time()

        if obs_in_file > 0:
            rnn_x = np.array(rnn_x)
            label_encodings = np.array(label_encodings)
            rnn_y = np.array(rnn_y)

            file_name = trainval + '_scanpaths_batch256.' + str(count_file)

            np.savez_compressed(os.path.join(save_path, file_name), rnn_x=rnn_x,
                                label_encodings=label_encodings, rnn_y=rnn_y)


            print("***** SAVED SMALLER FILE " + str(count_file) + " *****")
            print("Last len:", obs_in_file)
            obs_in_file = 0
            count_file += 1
            endf_time = time.time()
            print(timedelta(seconds=endf_time - initf_time))
            initf_time = time.time()

    last_time = time.time()
    print("TOTAL TIME: ")
    print(timedelta(seconds=last_time - first_time))
    local_time = time.ctime(last_time)
    print("Local time:", local_time)









