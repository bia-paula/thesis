from tkinter import font
#import data
from time import time
import numpy as np
import matplotlib.pyplot as plt
import rnn
from tensorflow.keras import callbacks
import rnn_afonso
from glob import glob


def cross_validate(K, model_function, vr, **kwargs):
    histories = []
    start = time()
    for k in range(K):
        X = rnn.BatchSequence(k, val=False, meu=1)
        print("Training data loaded")
        X_val = rnn.BatchSequence(k, val=True, meu=1)
        print("Validation data loaded")
        n_labels = X[0][0][1].shape[1]
        print(n_labels)
        model = model_function(n_labels)
        model.summary()
        histories.append(model.fit(X,
                                   validation_data=X_val,  # feed in the test data for plotting
                                   **kwargs).history)
        model.save('/Users/beatrizpaula/Downloads/' + vr + '_' + str(k) + '.h5')
        # plot_histories(histories, vr+str(k))
        print(f'Version {vr}, model {k}')
        print(f"Accuracy: {histories[k]['val_categorical_accuracy'][-1]}")

    print(f'Train time: {time() - start}')

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
    vr = "test4Jul"
    print(vr)
    #model_func = rnn.create_model
    model_func = rnn.create_model
    cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
    histories = cross_validate(K=1, model_function=model_func, vr=vr, epochs=10, callbacks=[cb])

    #X = rnn.BatchSequence(5, val=False)
    #TEST = X[0][1]
    #metrics = ['categorical_accuracy', 'val_categorical_accuracy', 'loss', 'val_loss']

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