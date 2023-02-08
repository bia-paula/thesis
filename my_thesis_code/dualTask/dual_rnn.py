import os.path

import numpy as np
from glob import glob

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, TimeDistributed, ConvLSTM2D, Flatten, BatchNormalization, \
    Reshape, Dropout, Multiply, MaxPool3D, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import random
import tensorflow as tf


import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.gaussianFPGT import onehotrnny2gaussian2d
from dataPreprocessing.foveateImages import get_circular_hard_foveate_dcb, gridcoord2ind

import config

global_data_path = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated"  # <-----------------  sequences by n fixations
#global_data_path = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/batchSize256"


# Load fp data as Sequence for keras
# meu = 0 for data prom Afonso instead of coco18
# batchSize selects folder for data with batches of that size
class BatchSequence(Sequence):
    # val = True for validation
    def __init__(self, k, val, batchSize, rnn_y_normal, label_encoding_heatmap=1, split=1, fovea_size=100, hs=0,
                 hs_first=0,
                 panoptic=0, hard_panoptic=None, accum=0, r=1, data_path=global_data_path):
        if hard_panoptic is None:
            hard_panoptic = [0, 0]
        self.rnn_y_normal = rnn_y_normal
        self.label_encoding_heatmap = label_encoding_heatmap
        '''if split:
            if not val:
                self.arr = np.setdiff1d(
                    np.array(glob("/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize"
                                  + str(batchSize) + "/train_scanpaths_fov100_batch" + str(batchSize) + "*")),
                    np.array(glob("/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize"
                                  + str(batchSize) + "/train_scanpaths_fov100_batch" + str(batchSize) + "." + str(
                        k) + '*')))
            else:
                self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize"
                                         + str(batchSize) + "/train_scanpaths_fov100_batch" + str(
                    batchSize) + "." + str(k) + "*"))
        else:'''
        if hard_panoptic[0]:
            if not val:
                p = os.path.join(data_path, "batchSize" + str(256), "train*")

            else:
                p = os.path.join(data_path, "batchSize" + str(256), "valid*")

        elif not hs:
            if not val:
                ''' p = [os.path.join(data_path, "train_scanpaths_fov" + str(fovea_size) + "*")]
                p.append(os.path.join(data_path, "TA", "train_scanpaths_fov" + str(fovea_size) +"*"))'''
                p = [os.path.join(data_path, "batchSize256", "train_scanpaths_fov" + str(fovea_size) + "*")]
                p.append(os.path.join(data_path, "batchSize256", "TA", "train_scanpaths_fov" + str(fovea_size) + "*"))

            else:
                '''p = [os.path.join(data_path, "valid_scanpaths_fov" + str(fovea_size) + "*")]
                p.append(os.path.join(data_path, "TA", "valid_scanpaths_fov" + str(fovea_size) + "*"))'''
                p = [os.path.join(data_path, "batchSize256", "valid_scanpaths_fov" + str(fovea_size) + "*")]
                p.append(os.path.join(data_path, "batchSize256","TA", "valid_scanpaths_fov" + str(fovea_size) + "*"))


        else:
            if not val:
                if hs_first:
                    p = os.path.join(data_path, "halfSequence", "*", "train" + "*" + "fov" +
                                     str(fovea_size) + "*")
                else:
                    p = os.path.join(data_path, "halfSequence", "second", "train" + "*" + "fov" +
                                     str(fovea_size) + "*")
            else:
                if hs_first:
                    p = os.path.join(data_path, "halfSequence", "*", "valid" + "*" + "fov" +
                                     str(fovea_size) + "*")
                else:
                    p = os.path.join(data_path, "halfSequence", "second", "valid" + "*" + "fov" +
                                     str(fovea_size) + "*")

        #self.arr = np.array(glob(p))
        print(len(np.array(glob(p[0]))))
        print(len(np.array(glob(p[1]))))

        self.arr = np.append(np.array(glob(p[0])), np.array(glob(p[0])), axis=0)
        np.random.shuffle(self.arr)

        self.hs_first = hs_first
        self.panoptic = panoptic
        self.hard_panoptic = hard_panoptic
        self.accum = accum
        self.r = r

    def __len__(self):
        return self.arr.size

    def __getitem__(self, idx):
        path = self.arr[idx]
        return load_batch(path, self.rnn_y_normal, self.label_encoding_heatmap, self.hs_first, self.panoptic,
                          self.hard_panoptic, self.accum, self.r)


# Loads rnn_y depending on fp ground truth encoding
def load_rnn_y(data, rnn_y_normal, path):
    rnn_y_fix = data['rnn_y']
    if rnn_y_normal:
        rnn_y_fix = np.apply_along_axis(onehotrnny2gaussian2d, 2, rnn_y_fix, sigma=1)

    detect_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/detect"                     #Change detect dir

    prev_name = path.split("/")[-1]
    id = ".".join(prev_name.split(".")[-3:])
    train_val = prev_name.split("_")[0]
    new_name = train_val + "_detected_batch256." + id

    if "TA" not in path:
        detect_path = os.path.join(detect_dir, new_name)
        rnn_y_det = np.load(detect_path)["rnn_y_det_binary"]
    else:
        rnn_y_det = np.zeros((rnn_y_fix.shape[0]), config.sequence_len, 1)


    '''if rnn_y_normal:
        batchSizeVal = path.split("batchSize")[-1].split("/")[0]
        normal_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + batchSizeVal
        split_path = path.split("/")
        path_rnn_y = normal_dir + "/normalFPGT/" + split_path[-1]
        path_rnn_y = path_rnn_y.replace("fov50", "fov100").replace("fov75", "fov100")
        rnn_y = np.load(path_rnn_y)['rnn_y_normal']
    else:
        rnn_y = data['rnn_y']'''

    return rnn_y_fix, rnn_y_det


# Loads rnn_y depending on fp ground truth encoding
def load_label_encoding(data, path, label_encoding_heatmap, panoptic):
    if label_encoding_heatmap:
        split_path = path.split("/")
        if label_encoding_heatmap == 2:
            name = split_path[-1] + "_fmapSize.npz"
        elif label_encoding_heatmap == 1:
            name = split_path[-1]
        path_label_enc = "/".join(split_path[:-1]) + "/heatmapLEGT/" + name
        path_label_enc = path_label_enc.replace("fov50", "fov100").replace("fov75", "fov100")
        label_enc = np.load(path_label_enc)['label_encodings_heatmap']
        if panoptic: label_enc = label_enc[:, :, :config.fmap_size_panoptic[2]]
    else:
        label_enc = data['label_encodings']
        label_enc = np.repeat(
            np.repeat(np.repeat(label_enc[:, np.newaxis, np.newaxis, np.newaxis], config.sequence_len, axis=1),
                      config.fmap_size[0], axis=2), config.fmap_size[1], axis=3)
    return label_enc


# hard_panoptic: [blur, fovea_radius]
def load_rnn_x(data, hard_panoptic, accum=0, r=1, rnn_y=None):
    if hard_panoptic[0]:
        dicts_list = data['rnn_x']
        rnn_x = []
        for idx, obs in enumerate(dicts_list):
            x0, y0 = np.array([config.fmap_size_panoptic[1], config.fmap_size_panoptic[0]]) // 2
            ind = np.full(config.sequence_len, gridcoord2ind(x0, y0))
            ind[1:] = np.argmax(rnn_y[idx], axis=1)[:-1]
            hr = obs['H']
            l_key = "L" + str(hard_panoptic[1])
            lr = obs[l_key]
            rnn_x.append(get_circular_hard_foveate_dcb(ind, hr, lr, r=r, accumulate=accum))

        rnn_x = np.array(rnn_x)

    else:
        rnn_x = data['rnn_x']

    return rnn_x


def load_batch(path, rnn_y_normal, label_encoding_heatmap, left, panoptic, hard_panoptic, accum, r):
    if hard_panoptic[0]:
        allow = True
    else:
        allow = False
    with np.load(path, allow_pickle=allow) as data:
        rnn_y_fix, rnn_y_det = load_rnn_y(data, rnn_y_normal, path)
        label_enc = load_label_encoding(data, path, label_encoding_heatmap, panoptic)
        # print("rnn_x: ", data['rnn_x'].shape, " label_enc: ", label_enc.shape, )

        rnn_x = load_rnn_x(data, hard_panoptic, accum, r, rnn_y_fix)

        '''if left:
            rnn_x = pad_sequences(rnn_x, dtype="float64", truncating="post", maxlen=4)
            rnn_y = pad_sequences(rnn_y, dtype="float64", truncating="post", maxlen=4)'''

        return [rnn_x, label_enc], [rnn_y_fix, rnn_y_det]

zeros_count = 214882
ones_count = 96049

def create_weighted_binary_crossentropy(zero_weight=0.3, one_weight=0.7):
    def weighted_binary_crossentropy(y_true, y_pred):
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        zero_weight = (zeros_count + ones_count) / (2 * zeros_count)
        one_weight = (zeros_count + ones_count) / (2 * ones_count)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

class TimeStepIDX(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def call(self, input_data):
        ts = np.array([0, 1, 0])

        idx = tf.shape(input_data[1])
        aux = tf.math.multiply(idx, ts)
        idx = int(tf.math.reduce_sum(aux))


        batch_size= int(tf.math.reduce_sum(input_data[2]))
        combinedInput = tf.slice(input_data[0], [0, 0, 0, 0, 0],
                                 [batch_size, idx, 320, 512, 512])
        return combinedInput



def create_model(task_vector_size, label_encoding_heatmap=0, time_steps=None, image_height=config.fmap_size[0],
                 image_width=config.fmap_size[1], channels=config.fmap_size[2],
                 output_size=config.fmap_size[0] * config.fmap_size[1], w_fix=0.5, fix_first=1,det_output_size=19):  # 2048

    img_seqs = Input(shape=(time_steps, image_height, image_width, channels))
    task = Input(shape=(task_vector_size,))
    # ftask = Dense(1000, activation = 'relu')(task)
    offsets = Dense(channels, activation='tanh')(task)
    offsets = Dropout(.4)(offsets)
    # offsets = Reshape(target_shape = (channels,1,1))(offsets)
    combinedInput = Multiply()([img_seqs, offsets])
    # combinedInput = BatchNormalization()(combinedInput) #new

    if fix_first:
        # Fixation First
        y_fix, state_h, state_c = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True,
                                             return_state=True,
                                             activation='relu')(combinedInput)
        y_det = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, activation='relu') \
            (combinedInput, initial_state=[state_h, state_c])

    else:
        # Detection First
        y_det, state_h, state_c = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, return_state=True,
                                             activation='relu')(combinedInput)
        y_fix = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, activation='relu')\
            (combinedInput, initial_state=[state_h, state_c])


    #y_det = BatchNormalization()(y_det)
    y_det = Dropout(0.5)(y_det)

    #z_det = TimeDistributed(Flatten())(y_det)
    out_det = Dense(512, activation='relu')(y_det)
    out_det = Dropout(0.5)(out_det)
    out_det = Dense(256, activation='relu')(out_det)
    out_det = Dropout(0.5)(out_det)
    out_det = Dense(1, activation='sigmoid', name='output_det')(out_det)

    # y = Dropout(.2)(y) #new
    y_fix = BatchNormalization()(y_fix)
    z_fix = TimeDistributed(Flatten())(y_fix)
    out_fix = Dense(output_size, activation='softmax', name='output_fix')(z_fix)

    model_rnn = Model([img_seqs, task], outputs=[out_fix, out_det])

    losses = {'output_fix': 'categorical_crossentropy',
              'output_det': create_weighted_binary_crossentropy(zero_weight=0.3, one_weight=0.7)}
    loss_weights = {'output_fix': w_fix, 'output_det': 1.0 - w_fix}

    metrics = {'output_fix': 'categorical_accuracy',
              'output_det': [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]}
    model_rnn.compile(loss=losses, loss_weights=loss_weights, optimizer=Adam(learning_rate=1e-3),
                      metrics=metrics)  # rmsprop
    model_rnn.summary()
    return model_rnn


if __name__ == "__main__":
    #rnn = create_model(9)
    #rnn.summary()
    #plot_model(rnn, to_file='model_plotConcat.png', show_shapes=True, show_layer_names=True)
    '''rnn_path = "/Volumes/DropSave/Tese/trainedModels/dual/Concat/fov50_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_10/fov50_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_10.h5"
    rnn3 = load_model(rnn_path, custom_objects={'weighted_binary_crossentropy': create_weighted_binary_crossentropy() })
    plot_model(rnn3, to_file='dual3.png', show_shapes=True, show_layer_names=True)
    rnn_path = "/Volumes/DropSave/Tese/trainedModels/dual/fov50_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_detectFirst_binary_dense64_dense32_wfix_25/fov50_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_detectFirst_binary_dense64_dense32_wfix_25.h5"
    rnn2 = load_model(rnn_path, custom_objects={'weighted_binary_crossentropy': create_weighted_binary_crossentropy()})
    plot_model(rnn2, to_file='dual2.png', show_shapes=True, show_layer_names=True)
    rnn_path = "/Volumes/DropSave/Tese/trainedModels/dual/fov50_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_10/fov50_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_10.h5"
    rnn1 = load_model(rnn_path, custom_objects={'weighted_binary_crossentropy': create_weighted_binary_crossentropy()})
    plot_model(rnn1, to_file='dual1.png', show_shapes=True, show_layer_names=True)
    rnn2.summary()'''
    rnn = create_model(18)
    rnn.summary()
