import os.path

import numpy as np
from glob import glob

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, TimeDistributed, ConvLSTM2D, Flatten, BatchNormalization, \
    Reshape, Dropout, Multiply, MaxPool3D, Concatenate, Conv3D, UpSampling3D, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.regularizers import L1
import random

import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.gaussianFPGT import onehotrnny2gaussian2d, onehotrnny2distdamping
from dataPreprocessing.foveateImages import get_circular_hard_foveate_dcb, gridcoord2ind

import config

global_data_path = "/Volumes/DropSave/Tese/dataset/sequences_by_nfixations"  # <-----------------  sequences by n fixations


# Load fp data as Sequence for keras
# meu = 0 for data prom Afonso instead of coco18
# batchSize selects folder for data with batches of that size
class BatchSequence(Sequence):
    # val = True for validation
    def __init__(self, k, val, batchSize, rnn_y_normal, label_encoding_heatmap, fovea_size=100, hs=0,
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
        if hard_panoptic:
            if not val:
                p = os.path.join(data_path, "batchSize" + str(256), "train*")

            else:
                p = os.path.join(data_path, "batchSize" + str(256), "valid*")

        elif not hs:
            if not val:
                p = os.path.join(data_path, "batchSize" + str(256), "train_scanpaths_fov" + str(fovea_size) + "*")

            else:
                p = os.path.join(data_path, "batchSize" + str(256), "valid_scanpaths_fov" + str(fovea_size) + "*")
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

        self.arr = np.array(glob(p))
        #np.random.shuffle(self.arr)
        self.arr.sort()
        print(self.arr[0])

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


# Loads rnn_y depending on fp ground truth encoding --------------------------------------------------------------
def load_rnn_y(data, rnn_y_normal):
    rnn_y = data['rnn_y']
    if rnn_y_normal:
        rnn_y = np.apply_along_axis(onehotrnny2gaussian2d, 2, rnn_y, sigma=1)
        #rnn_y = np.apply_along_axis(onehotrnny2distdamping, 2, rnn_y)

    '''if rnn_y_normal:
        batchSizeVal = path.split("batchSize")[-1].split("/")[0]
        normal_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + batchSizeVal
        split_path = path.split("/")
        path_rnn_y = normal_dir + "/normalFPGT/" + split_path[-1]
        path_rnn_y = path_rnn_y.replace("fov50", "fov100").replace("fov75", "fov100")
        rnn_y = np.load(path_rnn_y)['rnn_y_normal']
    else:
        rnn_y = data['rnn_y']'''

    return rnn_y


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
        label_enc = np.repeat(np.repeat(np.repeat(label_enc[:, np.newaxis, np.newaxis, np.newaxis], config.sequence_len, axis=1),
                              config.fmap_size_panoptic[0], axis=2), config.fmap_size_panoptic[1], axis=3 )
    return label_enc


# hard_panoptic: [blur, fovea_radius]
def load_rnn_x(data, hard_panoptic=0, accum=1, r=1, rnn_y=None):
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
            rnn_x.append(get_circular_hard_foveate_dcb(ind, hr, lr, r=r))


        rnn_x = np.array(rnn_x, dtype="float64")

    else:
        rnn_x = data['rnn_x']

    return rnn_x


def load_batch(path, rnn_y_normal, label_encoding_heatmap, left, panoptic, hard_panoptic, accum, r):
    if hard_panoptic[0]:
        allow = True
    else:
        allow = False
    with np.load(path, allow_pickle=allow) as data:
        rnn_y = load_rnn_y(data, rnn_y_normal)
        label_enc = load_label_encoding(data, path, label_encoding_heatmap, panoptic)
        # print("rnn_x: ", data['rnn_x'].shape, " label_enc: ", label_enc.shape, )

        rnn_x = load_rnn_x(data, hard_panoptic, accum, r, rnn_y)

        '''if left:
            rnn_x = pad_sequences(rnn_x, dtype="float64", truncating="post", maxlen=4)
            rnn_y = pad_sequences(rnn_y, dtype="float64", truncating="post", maxlen=4)'''
        return [rnn_x, label_enc], rnn_y

metrics = ['categorical_accuracy']
#metrics = ['categorical_accuracy', 'mean_squared_error', 'mean_absolute_error']

def create_model(task_vector_size, label_encoding_heatmap=0, time_steps=None, image_height=config.fmap_size[0],
                 image_width=config.fmap_size[1], channels=config.fmap_size[2], depth=1, sigmoid=0):

    img_seqs = Input(shape=(time_steps, image_height, image_width, channels), dtype="float64")
    img_seqs_in = img_seqs

    if label_encoding_heatmap == 2:
        task = Input(shape=(image_height, image_width, channels))
        combinedInput = Multiply()([img_seqs_in, task])
    else:
        task = Input(shape=(time_steps, image_height, image_width, task_vector_size))
        # ftask = Dense(1000, activation = 'relu')(task)
        offsets = Dense(channels, activation='tanh')(task)
        offsets = Dropout(.5)(offsets)
        # offsets = Reshape(target_shape = (channels,1,1))(offsets)
        combinedInput = Multiply()([img_seqs_in, offsets])
        #combinedInput = Concatenate(axis=-1)([img_seqs_in, task])

    conv = combinedInput

    y = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, activation='relu')(combinedInput)
    z = TimeDistributed(Flatten())(y)
    out = Dense(image_height*image_width, activation='softmax', name='output')(z)

    '''for d in range(depth):
        conv = ConvLSTM2D(filters=10, kernel_size=3, padding='same', return_sequences=True)(
            conv)
        conv = BatchNormalization()(conv)

    if sigmoid:
        out = Conv2D(filters=1, kernel_size=(2, 2), padding='same', activation='sigmoid')
        out = TimeDistributed(out)(conv)
        out = TimeDistributed(Flatten())(out)

    else:
        out = Conv3D(filters=1, kernel_size=(2, 2, 2), padding='same', activation='relu')
        out = TimeDistributed(Flatten())(out)
        out = Dense(units=160, activation='softmax')(out)'''

    '''embedding = Dense(units=152, activation='relu')(combinedInput)
    embedding = Dropout(rate=0.2)(embedding)

    embedding = TimeDistributed(Conv2D(filters=80, kernel_size=(2,2), padding='same', activation='relu'))(embedding)
    embedding = BatchNormalization()(embedding)



    conv = ConvLSTM2D(filters=20, kernel_size=2, padding='same', return_sequences=True, activation='relu')(conv)
    conv = BatchNormalization()(conv)

    conv = ConvLSTM2D(filters=10, kernel_size=2, padding='same', return_sequences=True, activation='relu')(conv)
    conv = BatchNormalization()(conv)'''

    #Experiment depth of convLSTM
    '''input_conv = combinedInput
    if not maintain_size:
        input_conv = ConvLSTM2D(filters=5, kernel_size=4, strides=2, return_sequences=True, activation='relu')(
            input_conv)
        input_conv = BatchNormalization()(input_conv)
        convs -= 1

    print("Convs: ", convs)

    for i in range(convs):
        print(i, "-"),
        input_conv = ConvLSTM2D(filters=5, kernel_size=4, padding='same', return_sequences=True, activation='relu')(input_conv)
        input_conv = BatchNormalization()(input_conv)
    print()
    output_convs = input_conv'''

    '''out = Conv2D(filters=1, kernel_size=(2, 2), padding='same', activation='sigmoid')#(output_convs)
    out = TimeDistributed(out)(conv)
    out = TimeDistributed(Flatten())(out)
    out = Softmax()(out)'''

    #out = TimeDistributed(Flatten())(output_convs)
    #out = Dense(units=output_size, activation="softmax")(out)

    model_rnn = Model([img_seqs, task], out)
    #metrics = [Precision(), Recall()]
    #loss KLDivergence
    #kullback_leibler_divergence categorical_crossentropy

    model_rnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                      metrics=metrics)  # rmsprop

    model_rnn.summary()
    print(model_rnn.loss)
    return model_rnn


if __name__ == "__main__":
    m = create_model(18, label_encoding_heatmap=1, image_height=config.fmap_size_panoptic[0],
                     image_width=config.fmap_size_panoptic[1], channels=config.fmap_size_panoptic[2],
                     output_size=config.fmap_size[0] * config.fmap_size[1], panoptic_sf_max=1)
    m.summary()
    plot_model(m, "RNN_panopticsf_max.png", show_shapes=True, show_layer_names=True)
