from tkinter import font
# import data
import time
import numpy as np
import matplotlib.pyplot as plt
# import rnn_panoptic_padded
import rnn
from tensorflow.keras import callbacks
from datetime import timedelta
import tensorflow as tf

from glob import glob
from keras import backend as K
from train import cross_validate, half_sequence

import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config

if __name__ == "__main__":

    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    K.clear_session()

    fovea_size = 100
    hs = 1

    for batch_size in [256]:
        data_path = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + str(batch_size)
        for rnn_y_normal in [1]:
            for label_encoding_heatmap in [0]:
                for hs_first in [1, 0]:
                    vr = "fov" + str(fovea_size) + "_batch" + str(batch_size) + config.rnn_y_type[rnn_y_normal] + \
                         config.label_encoding_type[label_encoding_heatmap] + "_halfSequence"
                    print(vr)
                    print(half_sequence[hs_first])
                    model_func = rnn.create_model
                    # model_func = rnn_afonso.create_model
                    cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

                    first_time = time.time()
                    local_time = time.ctime(first_time)
                    print("Local time:", local_time)

                    histories = cross_validate(K=5, model_function=model_func, vr=vr, epochs=100, batchSize=batch_size,
                                               rnn_y_normal=rnn_y_normal, label_encoding_heatmap=label_encoding_heatmap,
                                               fovea_size=fovea_size, hs=hs, hs_first=hs_first, data_path=data_path, callbacks=[cb])

                    last_time = time.time()
                    time_d = timedelta(seconds=last_time - first_time)
                    print("TOTAL TIME: ")
                    print(time_d)
                    local_time = time.ctime(last_time)
                    print("Local time:", local_time)

                    sfilename = "/Volumes/DropSave/Tese/trainedModels/" + vr + "/" + vr + half_sequence[hs_first] + "_histories"
                    tfilename = "/Volumes/DropSave/Tese/trainedModels/" + vr + "/" + vr + half_sequence[hs_first] + "_times.npy"
                    np.savez_compressed(sfilename, histories=histories)
                    d1 = {"time": time_d}
                    np.save(tfilename, d1)
                    K.clear_session()

