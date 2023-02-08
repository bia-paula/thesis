import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, ConvLSTM2D, Flatten, BatchNormalization, Dropout, Multiply
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/Code/RNN-visual-search-master/bia')

import config



# Load fp data as Sequence for keras
class BatchSequence(Sequence):
    # val = True for validation
    def __init__(self, k, val):
        if not val:
            #self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/train_scanpaths_fov100_*"))
            self.arr = np.array(["/Users/beatrizpaula/Downloads/exp_scanpaths_23jun.npz"])
        else:
            #self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/val_scanpaths_fov100_*"))
            self.arr = np.array(["/Users/beatrizpaula/Downloads/valid_exp_scanpaths_23jun.npz"])

    def __len__(self):
        return self.arr.size

    def __getitem__(self, idx):
        path = self.arr[idx]
        return load_batch(path)

def load_batch(path):
    with np.load(path) as data:
        return [data['rnn_x'], data['label_encodings']], data['rnn_y']

def create_model(task_vector_size, time_steps=None, image_height=10, image_width=16, channels=512, output_size=config.fmap_size[0] *
                                                                                                               config.fmap_size[1]):#2048

    img_seqs = Input(shape=(time_steps,image_height,image_width,channels))

    task = Input(shape=(task_vector_size,))
    #ftask = Dense(1000, activation = 'relu')(task)
    offsets = Dense(channels, activation='tanh')(task)
    offsets = Dropout(.4)(offsets)
    #offsets = Reshape(target_shape = (channels,1,1))(offsets)
    combinedInput = Multiply()([img_seqs,offsets])
    #combinedInput = BatchNormalization()(combinedInput) #new
    y = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, activation='relu')(combinedInput)
    #y = Dropout(.2)(y) #new
    y = BatchNormalization()(y)
    z = TimeDistributed(Flatten())(y)
    out = Dense(output_size, activation='softmax', name='output')(z)

    model_rnn = Model([img_seqs,task],out)

    model_rnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['categorical_accuracy'])#rmsprop

    return model_rnn

if __name__ == "__main__":

    m=create_model(18)
    m.summary()
    #plot_model(m, "myRNN_plo.png", show_shapes=True, show_layer_names=True)
