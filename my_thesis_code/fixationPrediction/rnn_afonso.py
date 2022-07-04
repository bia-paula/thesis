import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, TimeDistributed, ConvLSTM2D, Flatten, BatchNormalization, Reshape, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
import numpy as np
#import config

def create_model(task_vector_size, time_steps=None, image_height=12, image_width=12, channels=512, output_size=12**2):#2048

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
    model_rnn.summary()

    return model_rnn

if __name__ == "__main__":
    rnn = create_model(9)
    rnn.summary()