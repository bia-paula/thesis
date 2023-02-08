import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, ConvLSTM2D, Flatten, BatchNormalization, Dropout, Multiply, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/Code/RNN-visual-search-master/bia')
from fixationPrediction.rnn import load_batch

import config

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
    rng = np.random.default_rng()  # Create a default Generator.
    raw_inputs = []
    '''
    raw_inputs.append(np.random.rand(3, 10, 16, 256) * 400)
    raw_inputs.append(np.random.rand(5, 10, 16, 256) * 400)
    raw_inputs.append(np.random.rand(6, 10, 16, 256) * 400)
    
    raw_inputs.append(list(np.random.rand(3, 5, 8, 10) * 400))
    raw_inputs.append(list(np.random.rand(5, 5, 8, 10) * 400))
    raw_inputs.append(list(np.random.rand(6, 5, 8, 10) * 400))
    raw_inputs[0][0][ 0, 0, 0] = 0'''

    raw_inputs.append(np.random.rand(6, 10, 16, 256) * 400)
    raw_inputs.append(np.random.rand(6, 10, 16, 256) * 400)
    raw_inputs.append(np.random.rand(6, 10, 16, 256) * 400)

    raw_inputs = np.array(raw_inputs)

    padded = pad_sequences(raw_inputs, dtype="float64", truncating="post", maxlen=4, value=0.0)
    '''print(padded[0][0, 0, 0, 0])
    #print(padded[0][0][0, 0, 0])
    masking_layer = Masking(mask_value=0.0)

    masked_embedding = masking_layer(padded)
    y = ConvLSTM2D(filters=5, strides=2, kernel_size=4, return_sequences=True, activation='relu')(masked_embedding, mask=masked_embedding._keras_mask)

    y = BatchNormalization()(y)
    z = TimeDistributed(Flatten())(y)
    output_size = 40
    out = Dense(output_size, activation='softmax', name='output')(z)

    m = masked_embedding._keras_mask

    print(m[0])

    path = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/train_scanpaths_fov50_batch256.0.0.npz"
    out = load_batch(path, 0, 0)
    print(np.min(out[0][0]))
    print(np.max(out[0][0]))

    print("RAW")
    print("mean:", np.mean(raw_inputs))
    print("variance:", np.var(raw_inputs))

    print("Padded")
    print("mean:", np.mean(padded))
    print("variance:", np.var(padded))'''