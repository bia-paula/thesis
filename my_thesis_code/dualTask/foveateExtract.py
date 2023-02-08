import os.path

import laplacian_foveation as fv
import numpy as np
from PIL import Image, ImageDraw

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import math
import time

import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from datetime import timedelta
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt


def update_mask(mask, mask_pre):
    mask = mask + mask_pre
    return (mask >= 1).astype(np.float64)


# mask: (10, 16)
# hr, lr: (10, 16, 134)
def get_fmap_hard_panoptic(hr, lr, mask):
    mask = np.repeat(mask[:, :, np.newaxis], hr.shape[2], axis=2)
    m = (1 - mask) * lr + mask * hr
    return m


def ind2mask(x, y, r):
    seq_len = x.shape[0]
    Y, X = np.ogrid[:config.fmap_size_panoptic[0], :config.fmap_size_panoptic[1]]
    Y = np.repeat(np.expand_dims(Y, axis=2), seq_len, axis=2)
    X = np.repeat(np.expand_dims(X, axis=2), seq_len, axis=2)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    return mask.astype(np.float64)


# inds is array of size mem_size
def inds2mask(inds, r=1):
    # seq_len = x.shape[0]
    x, y = ind2gridcoord(inds)  # arrays of size mem_size
    Y, X = np.ogrid[:config.fmap_size_panoptic[0], :config.fmap_size_panoptic[1]]
    Y = np.expand_dims(Y, axis=0)
    X = np.expand_dims(X, axis=0)
    Y = np.repeat(Y, len(y), axis=0)
    X = np.repeat(X, len(x), axis=0)
    x, y = x[:, np.newaxis, np.newaxis], y[:, np.newaxis, np.newaxis]
    x = np.repeat(x, config.fmap_size_panoptic[1], axis=2)
    y = np.repeat(y, config.fmap_size_panoptic[0], axis=1)
    # Y = np.repeat(np.expand_dims(Y, axis=2), seq_len, axis=2)
    # X = np.repeat(np.expand_dims(X, axis=2), seq_len, axis=2)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r  # array of shape: (mem_size, 10, 16)
    return mask.astype(np.float64)


# Just for one FP
def gridcoord2mask(x, y, r=1, mem_size=20):
    Y, X = np.ogrid[:config.fmap_size_panoptic[0], :config.fmap_size_panoptic[1]]
    Y = torch.from_numpy(Y).unsqueeze(0).repeat((mem_size, 1, 1)).numpy()
    X = torch.from_numpy(X).unsqueeze(0).repeat((mem_size, 1, 1)).numpy()
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = (dist <= r).astype(np.float64)
    return mask


def get_circular_hard_foveate_dcb(ind, hr, lr, r=1, accumulate=1):
    px, py = ind2gridcoord(ind)
    mask = ind2mask(px, py, r)
    mask = torch.from_numpy(mask)
    mask = np.transpose(mask.unsqueeze(0).repeat(config.fmap_size_panoptic[-1], 1, 1, 1), (1, 2, 0, 3))
    hr = np.repeat(np.expand_dims(hr, axis=3), config.sequence_len, axis=3)
    lr = np.repeat(np.expand_dims(lr, axis=3), config.sequence_len, axis=3)
    if accumulate:
        for idx in range(1, len(ind)):
            mask[:, :, :, idx] += mask[:, :, :, idx - 1]
        mask = (np.array(mask >= 1)).astype(np.float64)
    dcb = (1 - mask) * lr + mask * hr

    return np.transpose(dcb, (3, 0, 1, 2))


def get_class_id(name):
    return config.classes.index(name)


def one_hot_encoder(name):
    enc = np.zeros(len(config.classes))
    enc[get_class_id(name)] = 1
    return enc


def smooth_foveate(img, x, y, fovea_size):
    # FOVEA SIZE
    sigma_xx = fovea_size
    sigma_yy = fovea_size
    sigma_xy = 0

    # PYRAMID LEVELS
    levels = 5

    height, width, channels = img.shape

    # Create the Laplacian blending object
    my_lap_obj = fv.LaplacianBlending(width, height, levels, sigma_xx, sigma_yy, sigma_xy)

    sigma_x = sigma_xx
    sigma_y = sigma_yy

    center = np.array([x, y])

    # FOVEA SIZE
    my_lap_obj.update_fovea(width, height, sigma_x, sigma_y, sigma_xy)

    return my_lap_obj.Foveate(img.astype(np.uint8), center.astype(int))


def get_classifier_features(fov_image, features_model):
    # Preprocess cnn input
    z = np.expand_dims(fov_image, axis=0)
    z = preprocess_input(z)
    # Extract features
    f_map = features_model.predict(z, verbose=False)
    return f_map


def init_panoptic_predictor():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    )
    model = build_backbone(cfg)
    model.eval()

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    model_coco = build_backbone(cfg)
    model_coco.eval()
    predictor = DefaultPredictor(cfg)
    return predictor


def get_panoptic_features(predictor, img_array):
    seg, info = predictor(img_array)["panoptic_seg"]
    feat = torch.zeros([80 + 54, 320, 512])
    for pred in info:
        mask = (seg == pred['id']).float()
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat[pred['category_id'] + 80, :, :] = mask
    return F.max_pool2d(F.interpolate(feat.unsqueeze(0), size=[20, 32]), [2, 2]).squeeze(0).permute(1, 2, 0).numpy()


def get_fmap(img_array, x, y, features_model, fovea_size, foveate_func):
    fov_image = foveate_func(img_array, x, y, fovea_size)
    f_map = get_classifier_features(fov_image, features_model)
    # f_map_pixel = get_panoptic_features(features_model, fov_image)

    return f_map
    # return f_map_pixel


def realcoord2gridcoord(real_x, real_y, resized=1,bbox_upleft=None):
    if resized:
        img_height = config.image_array_shape[0]
        img_width = config.image_array_shape[1]
    else:
        img_height = config.init_image_array_shape[0]
        img_width = config.init_image_array_shape[1]

    square_height = img_height / config.fmap_size[0]
    square_width = img_width / config.fmap_size[1]

    x = real_x // square_width
    y = real_y // square_height

    if bbox_upleft != 1:
        if x != 0 and real_x % square_width == 0:
            x -= 1
        if y != 0 and real_y % square_height == 0:
            y -= 1

    return int(x), int(y)


def gridcoord2realcoord(grid_x, grid_y):
    square_height = config.image_array_shape[0] / config.fmap_size[0]
    square_width = config.image_array_shape[1] / config.fmap_size[1]

    x = (grid_x + 0.5) * square_width
    y = (grid_y + 0.5) * square_height

    return x, y


# 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 #
# 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 # ----> top 3 rows of grid indexes
# 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 #


def gridcoord2ind(grid_x, grid_y):
    ind = grid_y * config.fmap_size[1] + grid_x
    return int(ind)


def ind2gridcoord(ind):
    y = ind // config.fmap_size[1]
    x = ind % config.fmap_size[1]
    return x, y


def fixated_object(fx, fy, x, y, w, h):
    if x <= fx <= x + w and y <= fy <= y + h:
        return True
    else:
        return False


if __name__ == '__main__':

    first_time = time.time()
    local_time = time.ctime(first_time)
    print("Local time:", local_time)
    # Load training data
    # path_fp_train = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
    # path_fp_train = "/Users/beatrizpaula/Downloads/human_scanpaths_TP_trainval_valid.json"
    path_fp = "/Volumes/DropSave/Tese/dataset/coco_search18_fixations_TA_trainval.json"  # <---------------------------------------------
    image_dir =  "/Volumes/DropSave/Tese/dataset/coco_search18_images_TA"

    vgg = VGG16(include_top=False, input_shape=config.image_array_shape)

    with open(path_fp) as fp:
        training_data = json.load(fp)

    # Set classifier model to extract features from
    fmap_model = VGG16(weights='imagenet', include_top=False, input_shape=config.image_array_shape)
    # fmap_model = init_panoptic_predictor()

    save_dir = '/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/TA'  # <--------------------------------------------------
    batch_size = 256
    new_img_size = config.original_image_size
    # Iterate for different fovea sizes
    for fovea_size in [100, 75, 50]:
        print("\t *** FOVEA SIZE", fovea_size, "***")

        # Extract from train and validation datasets
        for trainval in ["train", "valid"]:
            print("\t ***", trainval, "***")

            local_time = time.ctime(time.time())
            print("Local time:", local_time)

            # list of dictionaries with each experiment
            name_trainval = trainval

            count = 0
            count_files = 0

            # vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

            total_size = len(training_data)

            print("TOTAL SIZE: " + str(total_size))

            out = 0
            obs_in_file = []

            len_dict = {}

            for l in range(13):
                len_dict[l] = dict()
                len_dict[l]["rnn_x"] = np.zeros((256, l+1, 10, 16, 512))
                len_dict[l]["rnn_y"] = np.zeros((256, l+1, 160))
                len_dict[l]["label_encoding"] = np.zeros((256, 18))
                len_dict[l]["f_count"] = 0
                len_dict[l]["f_size"] = 0

            initf_time = time.time()

            for obs in training_data:
                init_obs = time.time()
                if obs["split"] != trainval:
                    count += 1
                    print(count, "trainval")
                    continue
                l = obs["length"] - 1
                if l >= 13:
                    count += 1
                    print(count, "length")
                    continue
                task = obs["task"]
                idx = len_dict[l]["f_size"]
                len_dict[l]["label_encoding"][idx] = one_hot_encoder(task) # np.zeros((256, 18))

                X = obs["X"]
                Y = obs["Y"]

                img_name = obs["name"]
                img = Image.open(os.path.join(image_dir, task, img_name))
                img_arr = img_to_array(img)

                for fp in range(obs["length"]):
                    fov_arr = smooth_foveate(img_arr, X[fp], Y[fp], fovea_size)
                    fov_img = array_to_img(fov_arr)
                    fov_img = fov_img.resize(new_img_size)
                    fov_arr = img_to_array(fov_img)
                    fmap = get_classifier_features(fov_arr, vgg)
                    len_dict[l]["rnn_x"][idx, fp] = fmap
                for fp in range(obs["length"]-1):
                    grid_x, grid_y = realcoord2gridcoord(X[fp+1], Y[fp+1], resized=0)
                    ind = gridcoord2ind(grid_x, grid_y)
                    len_dict[l]["rnn_y"][idx, fp, ind] = 1#np.zeros((256, 7, 160))

                grid_x, grid_y = realcoord2gridcoord(X[-1], Y[-1], resized=0)
                ind = gridcoord2ind(grid_x, grid_y)
                len_dict[l]["rnn_y"][idx, l, ind] = 1  # np.zeros((256, 7, 160))


                len_dict[l]["f_size"] += 1
                count += 1

                if len_dict[l]["f_size"] == batch_size:
                    file_name = trainval + "_scanpaths_fov" + str(fovea_size) + "_batch" + str(batch_size) + "_len" + \
                                str(l+1) + "." + str(len_dict[l]["f_count"])
                    np.savez_compressed(os.path.join(save_dir, file_name), rnn_x=len_dict[l]["rnn_x"],
                                        label_encoding=len_dict[l]["label_encoding"], rnn_y=len_dict[l]["rnn_y"])
                    len_dict[l]["f_count"] += 1
                    len_dict[l]["rnn_x"] = np.zeros((256, l+1, 10, 16, 512))
                    len_dict[l]["rnn_y"] = np.zeros((256, l+1, 160))
                    len_dict[l]["label_encoding"] = np.zeros((256, 18))
                    len_dict[l]["f_size"] = 0
                    print("SAVED FILE FOR LEN", l, "->",  len_dict[l]["f_count"]-1)

                end_obs = time.time()
                print(count, "/", len(training_data), "LEN", l+1, "(", len_dict[l]["f_size"], ")   ->",
                      timedelta(seconds=end_obs - init_obs))

            for l in range(13):
                if len_dict[l]["f_size"] > 0:
                    file_name = trainval + "_scanpaths_fov" + str(fovea_size) + "_batch" + str(batch_size) + "_len" + \
                                str(l+1) + "." + str(len_dict[l]["f_count"])
                    np.savez_compressed(os.path.join(save_dir, file_name), rnn_x=len_dict[l]["rnn_x"],
                                        label_encoding=len_dict[l]["label_encoding"], rnn_y=len_dict[l]["rnn_y"])
                    len_dict[l]["f_count"] += 1
                    len_dict[l]["rnn_x"] = np.zeros((256, l+1, 10, 16, 512))
                    len_dict[l]["rnn_y"] = np.zeros((256, l+1, 160))
                    len_dict[l]["label_encoding"] = np.zeros((256, 18))
                    len_dict[l]["f_size"] = 0
                    print("SAVED SHORTER FILE (", len_dict[l]["f_size"],")FOR LEN", l, "->",  len_dict[l]["f_count"]-1)

            endf_time = time.time()
            print("\nTIME ELAPSED:")
            print(timedelta(seconds=endf_time-initf_time), "\n")




