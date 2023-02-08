import json
from tkinter import font
# import data
import time
import numpy as np
import matplotlib.pyplot as plt
#import rnn_panoptic_padded
#import rnn
import dual_rnn as rnn
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model, model_from_config, model_from_json
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from glob import glob
from keras import backend as K

import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config

half_sequence = ["_second", "_first"]


def cross_validate(K, model_function, vr, batchSize,
                   rnn_y_normal, label_encoding_heatmap, fovea_size=100, hs=0, hs_first=0, panoptic=0,
                   hard_panoptic=None, r=1, data_path=rnn.global_data_path, convs=1, maintain_size=1, w_fix=0.5, fix_first=1,
                   **kwargs):
    if hard_panoptic is None:
        hard_panoptic = [0, 0]
    right_left = ""
    save_path = "/Volumes/DropSave/Tese/trainedModels/" + vr
    #save_path = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/" + vr
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

    n_labels = X[0][0][1].shape[-1]
    print(n_labels)
    print(X[0][0][0].shape)
    print(X[0][0][1].shape)
    print(X[0][1][0].shape)
    print(X[0][1][1].shape)

    if hard_panoptic[0]:
        fmap_size = config.fmap_size_panoptic
    else:
        fmap_size = config.fmap_size
    model = model_function(n_labels, image_height=fmap_size[0], image_width=fmap_size[1], channels=fmap_size[2],
                           w_fix=w_fix, det_output_size=n_labels+1, fix_first=fix_first)

    '''model = load_model("/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label_hardPanop_rpanop2_sig2_labelConcat_dense150_conv_f10/fov100_batch256_normal_rnny_onehot_label_hardPanop_rpanop2_sig2_labelConcat_dense150_conv_f10.h5")
    c = model.get_config()
    args = {"Sequence": rnn.BatchSequence}

    model = model.from_config(c)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                      metrics=['categorical_accuracy'])
    model.summary()'''

    cfg = model.get_config()
    with open(save_path + "/" + vr + '_cfg', "w") as fp:
        json.dump(cfg, fp)

    histories = model.fit(X, validation_data=X_val,  # feed in the test data for plotting
                          **kwargs).history
    #model.save(save_path + "/" + vr + half_sequence[hs_first] +'.h5')
    model.save(save_path + "/" + vr + '.h5')

    #metrics = rnn.metrics

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

def train():
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    K.clear_session()

    fovea_size = 100
    hard_panoptic = 0
    panoptic = 0
    r = 1

    models = []
    '''for maintain_size in [0, 1]:
    for convlstm in [5, 4, 3, 2]:
            print("\n", convlstm, "Convlstm's\n")'''
    for batch_size in [256]:
        for rnn_y_normal in [1]:
            for label_encoding_heatmap in [0]:
                for w_fix in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    for fix_first in [1]:
                        if fix_first:
                            first_name = "fixFirst_Concat_"
                        else:
                            first_name = "detectFirst"

                        vr = "fov" + str(fovea_size) + "_batch" + str(batch_size) + config.rnn_y_type[rnn_y_normal] + \
                             config.label_encoding_type[label_encoding_heatmap] +  \
                             "_dualLSTM_multiDetect_"+first_name+"_binary_dense64_dense32_wfix_" + str(w_fix*100)
                        # "_sig2_labelConcat_rnnyDamp_dense_f152_conv3_f40_cvlstm_f20_cvlstm_f10_tdf_dense_f160_soft"
                        print(vr)
                        model_func = rnn.create_model
                        # model_func = rnn_afonso.create_model
                        cb = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, restore_best_weights=True)]
                        #cb.append(callbacks.ModelCheckpoint(vr, save_best_only=True))

                        first_time = time.time()
                        local_time = time.ctime(first_time)
                        print("Local time:", local_time)

                        if hard_panoptic:
                            data_path = "/Users/beatrizpaula/Desktop/images_HL_DCBs/hardPanoptic"
                            hard_panoptic = [1, 2]
                        else:
                            data_path = rnn.global_data_path
                            hard_panoptic = None

                        histories = cross_validate(K=5, model_function=model_func, vr=vr, epochs=100, batchSize=batch_size,
                                                   rnn_y_normal=rnn_y_normal, label_encoding_heatmap=label_encoding_heatmap,
                                                   panoptic=panoptic, hard_panoptic=hard_panoptic, r=r, data_path=data_path,
                                                   w_fix=w_fix, callbacks=cb)

                        last_time = time.time()
                        time_d = timedelta(seconds=last_time - first_time)
                        print("TOTAL TIME: ")
                        print(time_d)
                        local_time = time.ctime(last_time)
                        print("Local time:", local_time)

                        sfilename = "/Volumes/DropSave/Tese/trainedModels/" + vr + "/" + vr + "_histories"
                        tfilename = "/Volumes/DropSave/Tese/trainedModels/" + vr + "/" + vr + "_times.npy"


                        #sfilename = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/" + vr + "/" + vr + "_histories"
                        #tfilename = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/" + vr + "/" + vr + "_times.npy"
                        np.savez_compressed(sfilename, histories=histories)
                        d1 = {"time": time_d}
                        np.save(tfilename, d1)
                        K.clear_session()

                        metrics = ["loss", "categorical_accuracy"]
                        # plot_histories(histories, vr+str(k))
                        print(f'Version {vr}')
                        for m in metrics:
                            for task in ["fix_", "det_"]:
                                task = 'output_' + task
                                print(m, histories['val_' + task + m][-1])

                        models.append(os.path.join("/Volumes/DropSave/Tese/trainedModels", vr, vr + '.h5'))
                        #models.append(os.path.join("/DATA/beatriz_cabarrao/beatriz_paula/dataset/trainedModels/", vr, vr + '.h5'))

                        del histories, d1, cb, model_func

    return models

if __name__ == "__main__":

    train()
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
