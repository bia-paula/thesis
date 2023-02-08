from tkinter import font
# import data
import time
import numpy as np
import matplotlib.pyplot as plt
import rnn_hardPanop as rnn
from tensorflow.keras import callbacks
from datetime import timedelta
import tensorflow as tf
import rnn_afonso
from glob import glob
from keras import backend as K

import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config


def cross_validate(K, model_function, vr, batchSize,
                   rnn_y_normal, label_encoding_heatmap, fovea_size=100, hs=0, hs_first=0, panoptic=0,
                   hard_panoptic=None, r=1, data_path=rnn.global_data_path, convs=1, maintain_size=1, **kwargs):
    if hard_panoptic is None:
        hard_panoptic = [0, 0]

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
                          panoptic=panoptic, hard_panoptic=hard_panoptic, accum=1, r=r, data_path=data_path)
    print("Training data loaded")

    X_val = rnn.BatchSequence(k, val=True, batchSize=batchSize, rnn_y_normal=rnn_y_normal,
                              label_encoding_heatmap=label_encoding_heatmap, fovea_size=fovea_size, split=0, hs=hs, hs_first=hs_first,
                              panoptic=panoptic, hard_panoptic=hard_panoptic, accum=1, r=r, data_path=data_path)
    print("Validation data loaded")

    n_labels = X[0][0][1].shape[1]
    print(n_labels)
    fmap_size = (10, 16, 134)
    model = model_function(n_labels, label_encoding_heatmap)
    model = model_function(n_labels, label_encoding_heatmap=label_encoding_heatmap, time_steps=None,
                           image_height=fmap_size[0], image_width=fmap_size[1], channels=fmap_size[2], depth=1,
                           sigmoid=0)
    model.summary()

    histories = model.fit(X, validation_data=X_val,  # feed in the test data for plotting
                          **kwargs).history
    model.save(save_path + "/" + vr + '_fmapSize.h5')
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

    label_encoding_heatmap = 0
    batch_size = 256
    rnn_y_normal = 1

    hard_panoptic = [1, 2]
    data_path = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/panoptic/hardPanoptic"
    accum = 1

    act = ["softm", "sigm"]

    for r in [1, 2, 3]:
        for sigmoid in [0, 1]:
            for rec_depth in [5, 3, 1]:

                vr = "hardPanop_sig" + str(hard_panoptic[1]) + "_r" + str(r) + "accum_normal_rnny_onehot_label_redDepth_" \
                     + str(rec_depth) + "_act_" + act[sigmoid]
                print(vr)
                model_func = rnn.create_model
                # model_func = rnn_afonso.create_model
                cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

                first_time = time.time()
                local_time = time.ctime(first_time)
                print("Local time:", local_time)

                histories = cross_validate(K=5, model_function=model_func, vr=vr, epochs=100, batchSize=batch_size,
                                           rnn_y_normal=rnn_y_normal, label_encoding_heatmap=label_encoding_heatmap,
                                           callbacks=[cb])

                histories = cross_validate(K=5, model_function=model_func, vr=vr, epochs=100, batchSize=batch_size,
                                           rnn_y_normal=rnn_y_normal, label_encoding_heatmap=label_encoding_heatmap,
                                           panoptic=1, hard_panoptic=hard_panoptic, r=r, data_path=data_path,
                                           depth=rec_depth, sigmoid=sigmoid,
                                           callbacks=cb)

                last_time = time.time()
                time_d = timedelta(seconds=last_time - first_time)
                print("TOTAL TIME: ")
                print(time_d)
                local_time = time.ctime(last_time)
                print("Local time:", local_time)

                sfilename = "/Volumes/DropSave/Tese/trainedModels/" + vr + "/" + vr + "_histories"
                tfilename = "/Volumes/DropSave/Tese/trainedModels/" + vr + "/" + vr + "_times.npy"
                np.savez_compressed(sfilename, histories=histories)
                d1 = {"time": time_d}
                np.save(tfilename, d1)
                K.clear_session()

    # X = rnn.BatchSequence(5, val=False)
    # TEST = X[0][1]
    # metrics = ['categorical_accuracy', 'val_categorical_accuracy', 'loss', 'val_loss']

    '''
    histories = np.load('../../plots/train_history/'+versions[0], allow_pickle=True)
    for vr in versions[1:]:
        new_hist = np.load('../../plots/train_history/'+vr, allow_pickle=True)
        for k in range(5):
            for m in metrics: 
                histories[k][m] = np.append(histories[k][m], new_hist[k][m]

    for vr in versions:
        histories = np.load('../../plots/train_history/' + vr, allow_pickle=True)

        # for k in range(5): print(np.argmin(histories[k]['val_loss']))

        scores, best1, best2 = get_results(histories)
        print(' '.join(map(str, scores)))
        # print(f'{scores} {best1} {best2}')
        # plot_histories([histories[best2]], versions[0]+'plot1'))
        '''