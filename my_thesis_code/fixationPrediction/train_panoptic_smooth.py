from tkinter import font
# import data
import time
import numpy as np
import matplotlib.pyplot as plt
#import rnn_panoptic_padded
import rnn
from tensorflow.keras import callbacks
from datetime import timedelta
import tensorflow as tf

from glob import glob
from keras import backend as K

import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config

half_sequence = ["_second", "_first"]


def cross_validate(K, model_function, vr, batchSize,
                   rnn_y_normal, label_encoding_heatmap, fovea_size=100, hs=0, hs_first=0, panoptic=0,
                   data_path=rnn.global_data_path, sf_max=0, **kwargs):

    save_path = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/" + vr
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    start = time.time()

    start_k = time.time()
    k = K

    X = rnn.BatchSequence(k, val=False, batchSize=batchSize, rnn_y_normal=rnn_y_normal,
                          label_encoding_heatmap=label_encoding_heatmap, fovea_size=fovea_size, split=0, hs=hs, hs_first=hs_first,
                          panoptic=panoptic, data_path=data_path)

    print("Training data loaded")

    X_val = rnn.BatchSequence(k, val=True, batchSize=batchSize, rnn_y_normal=rnn_y_normal,
                              label_encoding_heatmap=label_encoding_heatmap, split=0, hs=hs, hs_first=hs_first,
                              panoptic=panoptic, data_path=data_path)
    print("Validation data loaded")

    n_labels = X[0][0][1].shape[1]
    print(n_labels)
    if sf_max:
        fmap_size = config.fmap_size_panoptic_max
    else:
        fmap_size = config.fmap_size_panoptic

    model = model_function(n_labels, label_encoding_heatmap, image_height=fmap_size[0],
                               image_width=fmap_size[1], channels=fmap_size[2], panoptic_sf_max=sf_max)
    model.summary()

    histories = model.fit(X, validation_data=X_val,  # feed in the test data for plotting
                          **kwargs).history
    model.save(save_path + "/" + vr + half_sequence[hs_first] +'.h5')
    # plot_histories(histories, vr+str(k))
    print(f'Version {vr}')
    print(f"Accuracy: {histories['val_categorical_accuracy'][-1]}")
    print(f'Train time: {time.time() - start_k}')

    print(f'Train time: {time.time() - start}')
    del model

    return histories



def get_results(histories):
    acc1, acc2, loss = np.empty((len(histories))), np.empty((len(histories))), np.empty((len(histories)))

    for (i, model) in enumerate(histories):
        acc1[i] = model['val_categorical_accuracy'][-1]
        ind = np.argmin(model['val_loss'])
        loss[i] = model['val_loss'][ind]
        acc2[i] = model['val_categorical_accuracy'][ind]

    return np.array([np.mean(acc1), np.std(acc1), np.mean(acc2), np.std(acc2)]), np.argmax(acc1), np.argmin(loss)


def plot_histories(histories, version,
                   metrics=['categorical_accuracy', 'val_categorical_accuracy', 'loss', 'val_loss']):
    """
    function to plot the histories of data
    """
    '''fig, axes = plt.subplots(nrows = (len(metrics) - 1) // 2 + 1, ncols = 2, figsize = (16,16))
    axes = axes.reshape((len(metrics) - 1) // 2 + 1, 2)
    for i,metric in enumerate(metrics):
        for history in histories:
            axes[(i+2)//2 - 1, 1 - (i+1)%2].plot(history[metric])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].legend([i for i in range(len(histories))])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].set_xticks(
                np.arange(max(history[metric]))
            )'''

    fig, axes = plt.subplots(nrows=2, ncols=1)
    # axes = axes.reshape(2, len(histories))
    fig.set_size_inches(18, 12)

    for j, hist in enumerate(histories):
        # summarize history for accuracy
        axes[0].plot(hist['categorical_accuracy'])
        axes[0].plot(hist['val_categorical_accuracy'])
        # axes[0].set_title('model '+str(0)+' accuracy')
        axes[0].set_ylabel('Accuracy', fontsize=15)
        axes[0].tick_params(labelsize=12)
        # axes[0].legend(['train', 'test'], loc='upper right')

        # summarize history for loss
        axes[1].plot(hist['loss'])
        axes[1].plot(hist['val_loss'])
        # axes[1].set_title('model '+str(0)+' loss' )
        axes[1].set_ylabel('Loss', fontsize=15)
        axes[1].set_xlabel('Epoch', fontsize=15)
        axes[1].tick_params(labelsize=12)
        # axes[1].legend(['Train', 'Test'], loc='upper right')

        lgd = axes[0].legend(['Train', 'Test'], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                             fontsize=15)

    plt.savefig('../../plots/crossval_' + version, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    K.clear_session()

    fovea_size = 100
    data_dir = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/panoptic/IRL_smoothx"
    sf_labels = ["IRL", "MAX"]

    for batch_size in [256]:
        for rnn_y_normal in [1, 0]:
            for label_encoding_heatmap in [0, 1, 2]:
                for sf_max in [0,1]:
                    
                    vr = "fov" + str(fovea_size) + "_batch" + str(batch_size) + config.rnn_y_type[rnn_y_normal] + \
                         config.label_encoding_type[label_encoding_heatmap] + "_sfPanop_" + sf_labels[sf_max]
                    print(vr)
                    model_func = rnn.create_model
                    # model_func = rnn_afonso.create_model
                    cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

                    first_time = time.time()
                    local_time = time.ctime(first_time)
                    print("Local time:", local_time)

                    histories = cross_validate(K=5, model_function=model_func, vr=vr, epochs=100, batchSize=batch_size,
                                               rnn_y_normal=rnn_y_normal, label_encoding_heatmap=label_encoding_heatmap,
                                               sf_max=sf_max, data_path=data_dir, callbacks=[cb])

                    last_time = time.time()
                    time_d = timedelta(seconds=last_time - first_time)
                    print("TOTAL TIME: ")
                    print(time_d)
                    local_time = time.ctime(last_time)
                    print("Local time:", local_time)

                    sfilename = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/" + vr + "/" + vr + "_histories"
                    tfilename = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/" + vr + "/" + vr + "_times.npy"
                    np.savez_compressed(sfilename, histories=histories)
                    d1 = {"time": time_d}
                    np.save(tfilename, d1)
                    K.clear_session()


