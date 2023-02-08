import json
import time
from datetime import timedelta
import os
import glob
from sklearn.utils import class_weight

import sys

import numpy as np

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import fixated_object

if __name__ == "__main__":

    path_fp = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_"
    save_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/detect"

    positive = [0, 0]
    negative = [0, 0]

    for tv_idx, trainval in enumerate(["train.json", "valid.json"]):
        tv = trainval.split(".")[0]
        path_fp_train = path_fp + trainval
        fp = open(path_fp_train)
        training_data = json.load(fp)  # list of dictionaries with each experiment
        image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"
        name_trainval = trainval.split(".")[0]

        out = 0
        batch_size = 256
        file_size = 0
        rnn_y_det_binary = np.zeros((batch_size, config.sequence_len))
        rnn_y_det_multi = np.zeros((batch_size, config.sequence_len, len(config.classes) + 1))
        count = 0
        count_files = 0
        total = len(training_data)
        file_start = time.time()



        for idx_obs, obs in enumerate(training_data):

            flag = 0
            task = config.classes.index(obs["task"])

            # Check how many fp it took to detect object
            for i_fixated in range(min(obs["length"], 7)):
                if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1],
                                  obs["bbox"][2],
                                  obs["bbox"][3]):
                    flag = 1
                    break

            # Ignore observations that took longer than 6 fp to find object
            if not flag:
                out += 1
                print(out)
                count += 1
                continue

            for i_fixated in range(7):
                seq_i = i_fixated
                if i_fixated >= obs["length"]:
                    seq_i = obs["length"] - 1

                if fixated_object(obs["X"][seq_i], obs["Y"][seq_i], obs["bbox"][0], obs["bbox"][1],
                                  obs["bbox"][2],
                                  obs["bbox"][3]):
                    rnn_y_det_binary[file_size, i_fixated] = 1
                    rnn_y_det_multi[file_size, i_fixated, task] = 1
                    positive[tv_idx] += 1

                else:
                    rnn_y_det_multi[file_size, i_fixated, -1] = 1
                    negative[tv_idx] += 1

            file_size += 1
            count += 1
            print(count, "/", total)

            # Save File if batch complete
            if file_size == batch_size:
                print("In filse size")
                a = count_files // 15
                b = count_files % 15
                filename = tv + "_detected_batch"+str(batch_size)+"." + str(a) +"." + str(b)
                '''np.savez_compressed(os.path.join(save_dir, filename), rnn_y_det_binary=rnn_y_det_binary,
                                    rnn_y_det_multi=rnn_y_det_multi)'''
                count_files += 1
                file_size = 0
                rnn_y_det_binary = np.zeros((batch_size, config.sequence_len))
                rnn_y_det_multi = np.zeros((batch_size, config.sequence_len, len(config.classes) + 1))

                print("SAVE FILE", count_files)
                file_end = time.time()
                print(timedelta(seconds=file_end - file_start))
                file_start = time.time()

        if file_size > 0:
            a = count_files // 15
            b = count_files % 15
            filename = tv + "_detected_batch" + str(batch_size) + "." + str(a) + "." + str(b)
            rnn_y_det_binary = rnn_y_det_binary[:file_size]
            rnn_y_det_multi = rnn_y_det_multi[:file_size]
            '''np.savez_compressed(os.path.join(save_dir, filename), rnn_y_det_binary=rnn_y_det_binary,
                                rnn_y_det_multi=rnn_y_det_multi)'''
            count_files += 1
            file_size = 0

            print("SAVE LAST FILE", count_files)

            file_end = time.time()
            print(timedelta(seconds=file_end - file_start))


    print("Positive:", positive)
    print("Negative:", negative)




    '''for trainval, nfiles in [["train*", 74], ["valid*", 12]]:

        before_files = glob.glob(os.path.join(save_dir, trainval))

        for before in before_files:
            number = int(before.split(".")[-2])
            a = number // 15
            b = number % 15
            new = str(a) + "." + str(b) + ".npz"

            after = before.replace(str(number) + ".npz", new)
            os.rename(before, after)'''
