# Version 1 of DCBs FP
# For padded and truncated data to uniform length of 6

import numpy as np
import torch
from glob import glob

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, TimeDistributed, ConvLSTM2D, Flatten, BatchNormalization, \
    Reshape, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import ind2gridcoord, gridcoord2ind

import config

def ind2mask(x, y, r):
    Y, X = np.ogrid[:config.fmap_size[0], :config.fmap_size[1]]
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return mask.astype(np.float32)


def get_circular_hard_foveate_dcb(hr, lr, ind, r=2, accumulate=0):
    px, py = ind2gridcoord(ind)
    mask = ind2mask(px, py, r)
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).repeat(config.fmap_size_panoptic[-1], 1, 1)
    dcb = (1 - mask) * lr + mask * hr
    return dcb

## Fazer load do rnny e criar rnnx





# Load fp data as Sequence for keras
# meu = 0 for data prom Afonso instead of coco18
# batchSize selects folder for data with batches of that size
class BatchSequence(Sequence):
    # val = True for validation
    def __init__(self, k, val, batchSize, rnn_y_normal, label_encoding_heatmap, split=1):
        self.rnn_y_normal = rnn_y_normal
        self.label_encoding_heatmap = label_encoding_heatmap
        if split:
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
        else:
            if not val:
                self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/sequences_fixated_in_6_padded_truncated/panoptic/max/batchSize"
                                         + str(batchSize) + "/train*"))
            else:
                self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/sequences_fixated_in_6_padded_truncated/panoptic/max/batchSize"
                                         + str(batchSize) + "/valid*"))

    def __len__(self):
        return self.arr.size

    def __getitem__(self, idx):
        path = self.arr[idx]
        return load_batch(path, self.rnn_y_normal, self.label_encoding_heatmap)

# Loads rnn_y depending on fp ground truth encoding
def load_rnn_y(data, path, rnn_y_normal):
    if rnn_y_normal:
        batchSizeVal = path.split("batchSize")[-1].split("/")[0]
        normal_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + batchSizeVal
        split_path = path.split("/")
        path_rnn_y = normal_dir + "/normalFPGT/" + split_path[-1]
        path_rnn_y = path_rnn_y.replace("fov50", "fov100").replace("fov75", "fov100")
        rnn_y = np.load(path_rnn_y)['rnn_y_normal']
    else:
        rnn_y = data['rnn_y']
    return rnn_y

# Loads rnn_y depending on fp ground truth encoding
def load_label_encoding(data, path, label_encoding_heatmap):
    if label_encoding_heatmap:
        split_path = path.split("/")
        if label_encoding_heatmap == 2:
            name = split_path[-1] + "_fmapSize.npz"
        elif label_encoding_heatmap == 1:
            name = split_path[-1]
        path_label_enc = "/".join(split_path[:-1]) + "/heatmapLEGT/" + name
        path_label_enc = path_label_enc.replace("fov50", "fov100").replace("fov75", "fov100")
        label_enc = np.load(path_label_enc)['label_encodings_heatmap']
    else:
        label_enc = data['label_encodings']
    return label_enc


def load_batch(path, rnn_y_normal, label_encoding_heatmap):
    with np.load(path) as data:
        rnn_y = load_rnn_y(data, path, rnn_y_normal)
        label_enc = load_label_encoding(data, path, label_encoding_heatmap)
        #print("rnn_x: ", data['rnn_x'].shape, " label_enc: ", label_enc.shape, )
        return [data['rnn_x'], label_enc], rnn_y


def create_model(task_vector_size, label_encoding_heatmap=0, time_steps=None, image_height=config.fmap_size[0],
                 image_width=config.fmap_size[1], channels=config.fmap_size[2],
                 output_size=config.fmap_size[0] * config.fmap_size[1]):  # 2048

    img_seqs = Input(shape=(time_steps, image_height, image_width, channels))
    if label_encoding_heatmap == 2:
        task = Input(shape=(image_height, image_width, channels))
        combinedInput = Multiply()([img_seqs, task])
    else:
        task = Input(shape=(task_vector_size,))
        # ftask = Dense(1000, activation = 'relu')(task)
        offsets = Dense(channels, activation='tanh')(task)
        offsets = Dropout(.5)(offsets)
        # offsets = Reshape(target_shape = (channels,1,1))(offsets)
        combinedInput = Multiply()([img_seqs, offsets])

    # combinedInput = BatchNormalization()(combinedInput) #new

    y = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, activation='relu')(combinedInput)
    # y = Dropout(.2)(y) #new
    y = BatchNormalization()(y)
    z = TimeDistributed(Flatten())(y)
    out = Dense(output_size, activation='softmax', name='output')(z)

    model_rnn = Model([img_seqs, task], out)

    model_rnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                      metrics=['categorical_accuracy'])  # rmsprop

    return model_rnn


if __name__ == "__main__":
    m = create_model(18, "onehot")
    m.summary()
    # plot_model(m, "myRNN_plo.png", show_shapes=True, show_layer_names=True)
