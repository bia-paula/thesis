import json
import os.path
import random
import math

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.image import resize_with_pad
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from glob import glob
import sys
from os import mkdir
from os.path import join

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import fixated_object, smooth_foveate, one_hot_encoder, gridcoord2ind

from datetime import timedelta
from time import time, ctime

classSizes_path = "/Users/beatrizpaula/Desktop/Tese/my_thesis_code/detection/class_sizes.json"
image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"
save_train_data = "/Volumes/DropSave/Tese/dataset/detection_dataset"

sizes_true_train = [2026, 1433, 1098, 2143, 1126, 2833, 1985, 1605, 1495, 1305, 1701, 870, 1182, 1770, 2179, 1213, 1056, 3057]
sizes_true_valid = [314, 219, 214, 312, 199, 406, 285, 256, 214, 188, 264, 149, 184, 288, 374, 163, 168, 437]


def get_train_feature_maps(path, fovea, trainval, negative=0):
    file_start = time()

    if negative:
        true_false = "false"

    else:
        true_false = "true"

    with open(classSizes_path) as fp:
        sizes = json.load(fp)

    heights = sizes["height"]
    widths = sizes["width"]

    with open(path) as fp:
        data = json.load(fp)

    fov = "fov" + str(fovea)

    total_size = 0
    for d in data:
        total_size += len(d)

    try:
        mkdir(join(save_train_data, fov))
    except FileExistsError:
        pass

    progress = 0
    for c_idx, c in enumerate(config.classes):

        class_start = time()

        try:
            mkdir(join(save_train_data, fov, c))
        except FileExistsError:
            pass

        h, w = heights[config.classes.index(c)], widths[config.classes.index(c)]

        data_class = data[config.classes.index(c)]
        for count, fix in enumerate(data_class):
            x, y = fix["fp"]
            fv = loadFoveatedImage(fix["task"], fix["name"], x, y, fovea)
            img = cropImage(fv, x, y, fix["task"], heights, widths)

            img = resize_with_pad(img, int(h * 2), int(w * 2))
            I = array_to_img(img)
            idx = gridcoord2ind(x, y)
            name = true_false + trainval + "_" + fix["name"].split(".")[0] + "_" + str(count) + ".jpg"
            I.save(os.path.join(save_train_data, fov, c, name))
            progress += 1

            if progress % 50 == 0:
                print(progress, "/", total_size)

        print(c, "DONE!")
        class_end = time()
        print(timedelta(seconds=class_end - class_start))

    file_end = time()
    print(timedelta(seconds=file_end - file_start))


def createModel(h, w, by_class=0):
    if by_class:
        output = 1
        activ = 'sigmoid'
    else:
        output = len(config.classes) + 1 #Plus None classs
        activ = 'softmax'
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(h, w, 3))
    # Freeze four convolution blocks layers[:15]
    for layer in vgg_model.layers:
        layer.trainable = False

    x = vgg_model.output
    x = Flatten()(x)  # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output, activation=activ)(x)  # Softmax for multiclass
    transfer_model = Model(inputs=vgg_model.input, outputs=x)

    return transfer_model


def train(transfer_model, X, X_val, by_class=0):
    if by_class:
        l = "binary_crossentropy"
        c = 'accuracy'
    else:
        l = "categorical_crossentropy"
        c = 'categorical_accuracy'
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
    checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    learning_rate = 5e-5
    transfer_model.compile(loss=l, optimizer=Adam(lr=learning_rate),
                           metrics=[c])

    history = transfer_model.fit(X, batch_size=1, epochs=50, validation_data=X_val,
                                 callbacks=[lr_reduce, checkpoint])

    return history


class BatchSequence(Sequence):
    def __init__(self, path, batch_size, val, h, w, by_class=0, task=None):

        if by_class:
            path = os.path.join(path, task)
            if val:
                self.array = glob(path + "/*valid*")
            else:
                self.array = glob(path + "/*train*")
        else:
            #path = os.path.join(path, "*")
            self.get_true_false_samples(path, val)


        random.shuffle(self.array)
        self.batch_size = batch_size
        self.by_class = by_class
        self.h = h
        self.w = w


    def __len__(self):
        return math.ceil(float(len(self.array)) / float(self.batch_size))

    def __getitem__(self, idx):
        return self.load_batch(idx)

    def get_true_false_samples(self, path, val):
        tv = ["train*", "valid*"]
        train_val = tv[val]
        self.array = []
        for t_idx, task in enumerate(config.classes):
            task_path = os.path.join(path, task)
            self.array.extend(glob(task_path + "/true_" + train_val))
            #Select false samples
            falses = glob(task_path + "/false_" + train_val)
            random.shuffle(falses)
            if val:
                false_len = int(sizes_true_valid[t_idx]/len(config.classes))
            else:
                false_len = int(sizes_true_train[t_idx]/len(config.classes))
            self.array.extend(falses[:false_len])

    def get_gt(self, path):
        if self.by_class:
            if "true" in path:
                y = 1
            else:
                y = 0
        else:
            y = np.zeros(len(config.classes) + 1)
            if "true" in path:
                task = path.split("/")[-2]
                y[config.classes.index(task)] = 1
            else:
                y[-1] = 1

        return y

    def load_batch(self, idx):

        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.array))

        this_batch = end - start

        X = np.zeros((this_batch, self.h, self.w, 3))
        if self.by_class:
            Y = np.zeros(this_batch)
        else:
            Y = np.zeros((this_batch, len(config.classes) + 1))

        for i, path in enumerate(self.array[start:end]):
            img = Image.open(path)
            img_arr = img_to_array(img)
            img_arr = resize_with_pad(img_arr, self.h, self.w)
            x = preprocess_input(img_arr)
            y = self.get_gt(path)
            X[i] = x
            Y[i] = y

        return X, Y


def loadFoveatedImage(class_name, image_id, x, y, fovea):
    image_path = image_dir + "/" + class_name + "/" + image_id
    img = Image.open(image_path)
    img_array = img_to_array(img)
    return smooth_foveate(img_array, x, y, fovea)


def cropImage(img, x, y, task, heights, widths):
    c = config.classes.index(task)
    h = heights[c]
    w = widths[c]
    min_h, max_h = int(max(0, y - h)), int(min(config.original_image_size[1], y + h))
    min_w, max_w = int(max(0, x - w)), int(min(config.original_image_size[0], x + w))
    return img[min_h:max_h, min_w:max_w]


def get_train_data(train_val):
    train_path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval" + train_val + ".json"
    with open(train_path) as fp:
        data = json.load(fp)

    detected = []
    detected_by_classes = []

    for i in range(len(config.classes)):
        detected_by_classes.append([])

    for obs in data:
        fixes = []
        for i_fixated in range(min(obs["length"], 7)):
            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                fixes.append([obs["X"][i_fixated], obs["Y"][i_fixated]])

        if not len(fixes):
            continue

        class_name = obs["task"]
        image_id = obs["name"]

        for f in fixes:
            detected.append({"task": class_name, "name": image_id, "fp": f})
            detected_by_classes[config.classes.index(class_name)].append(
                {"task": class_name, "name": image_id, "fp": f})

    with open("fineTuneClassifier_" + train_val + ".json", "w") as fp:
        json.dump(detected, fp)

    with open("fineTuneClassifierByClasses_" + train_val + ".json", "w") as fp:
        json.dump(detected_by_classes, fp)

    print(len(detected))

    for i in range(len(config.classes)):
        print(config.classes[i], ":", len(detected_by_classes[config.classes.index(config.classes[i])]))


def get_negative_train_data(train_val):
    train_path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval" + train_val + ".json"
    with open(train_path) as fp:
        data = json.load(fp)

    detected = []
    detected_by_classes = []
    remain = []

    for i in range(len(config.classes)):
        detected_by_classes.append([])
        remain.append(0)

    for obs in data:
        fixes = []
        total = []
        for i_fixated in range(min(obs["length"], 7)):
            total.append([obs["X"][i_fixated], obs["Y"][i_fixated]])
            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                fixes.append([obs["X"][i_fixated], obs["Y"][i_fixated]])
                total.pop()

        if not len(fixes):
            continue

        class_name = obs["task"]
        image_id = obs["name"]

        random.shuffle(total)

        to_do = len(fixes) + remain[config.classes.index(class_name)]

        done = 0

        for f_idx in range(min(to_do, len(total))):
            f = total[f_idx]
            detected.append({"task": class_name, "name": image_id, "fp": f})
            detected_by_classes[config.classes.index(class_name)].append(
                {"task": class_name, "name": image_id, "fp": f})
            done += 1

        remain[config.classes.index(class_name)] = to_do - done

    with open("fineTuneClassifier_negative_" + train_val + ".json", "w") as fp:
        json.dump(detected, fp)

    with open("fineTuneClassifierByClasses_negative_" + train_val + ".json", "w") as fp:
        json.dump(detected_by_classes, fp)

    print(len(detected))

    for i in range(len(config.classes)):
        print(config.classes[i], ":", len(detected_by_classes[config.classes.index(config.classes[i])]))


if __name__ == '__main__':

    with open(classSizes_path) as fp:
        sizes = json.load(fp)

    fixations_dir = "/Users/beatrizpaula/Desktop/Tese/my_thesis_code/detection/fineTuneClassifierByClasses"

    for fovea in [100, 75, 50]:
        '''if fovea != 100:

            for i_tf, true_false in [[0, ""], [1, "_negative"]]:
                for train_val in ["_train", "_valid"]:
                    print("Local time:", ctime(time()))

                    fixations_path = fixations_dir + true_false + train_val + ".json"

                    get_train_feature_maps(fixations_path, fovea, train_val, negative=i_tf)'''


        path = "/Volumes/DropSave/Tese/dataset/detection_dataset/fov" + str(fovea)
        save_path = "/Volumes/DropSave/Tese/trainedModels/classifier/"
        batch_size = 32

        # Single class Classifiers
        '''if fovea == 100:
            for task_idx, task in enumerate(config.classes):
                model_start = time()
                h = int(2 * sizes['height'][task_idx])
                w = int(2 * sizes['width'][task_idx])

                vr = "fov" + str(fovea) + "_vgg16_" + task
                print(vr)

                model = createModel(h, w, by_class=1)

                model.summary()

                lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max',
                                              min_lr=5e-5)
                checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor='val_accuracy', mode='max', save_best_only=True,
                                             verbose=1)
                learning_rate = 1e-3
                model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate),
                              metrics=["accuracy"])

                X = BatchSequence(path, batch_size, 0, h, w, by_class=1, task=task)
                X_val = BatchSequence(path, batch_size, 1, h, w, by_class=1, task=task)

                histories = model.fit(X, validation_data=X_val,  # feed in the test data for plotting
                                      epochs=50, callbacks=[lr_reduce]
                                      ).history

                model.save(save_path + vr + '.h5')
                model_end = time()
                sfilename = "/Volumes/DropSave/Tese/trainedModels/classifier/" + vr + "_histories"
                tfilename = "/Volumes/DropSave/Tese/trainedModels/classifier/" + vr + "_times.npy"
                np.savez_compressed(sfilename, histories=histories)
                d1 = {"time": timedelta(seconds=(model_end - model_start))}
                np.save(tfilename, d1)
                '''
        #Multi Class classifier
        model_start = time()
        h = int(2 * max(sizes['height']))
        w = int(2 * max(sizes['width']))

        vr = "fov" + str(fovea) + "_vgg16"
        print(vr)

        model = createModel(h, w, by_class=0)
        model.summary()
        plot_model(model, to_file='model_plotClass.png', show_shapes=True, show_layer_names=True)

        lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max',
                                      min_lr=5e-5)
        checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor='val_accuracy', mode='max', save_best_only=True,
                                     verbose=1)
        learning_rate = 1e-3
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate),
                      metrics=["categorical_accuracy"])

        X = BatchSequence(path, batch_size, 0, h, w, by_class=0)
        X_val = BatchSequence(path, batch_size, 1, h, w, by_class=0)

        histories = model.fit(X, validation_data=X_val,  # feed in the test data for plotting
                              epochs=20, callbacks=[lr_reduce]
                              ).history

        model.save(save_path + vr + '.h5')
        model_end = time()
        sfilename = "/Volumes/DropSave/Tese/trainedModels/classifier/" + vr + "_histories"
        tfilename = "/Volumes/DropSave/Tese/trainedModels/classifier/" + vr + "_times.npy"
        np.savez_compressed(sfilename, histories=histories)
        d1 = {"time": timedelta(seconds=(model_end - model_start))}
        np.save(tfilename, d1)






