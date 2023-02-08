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
    return feat


def get_fmap(img_array, x, y, features_model, fovea_size, foveate_func):
    fov_image = foveate_func(img_array, x, y, fovea_size)
    #f_map = get_classifier_features(fov_image, features_model)
    f_map_pixel = get_panoptic_features(features_model, fov_image)

    nearest_interp = F.interpolate(f_map_pixel.unsqueeze(0), size=[20, 32]).squeeze(0).permute(1, 2, 0).numpy()

    max_pool = F.max_pool2d(f_map_pixel, (32, 32)).permute(1, 2, 0).numpy()

    #return f_map
    return nearest_interp, max_pool


def realcoord2gridcoord(real_x, real_y, bbox_upleft=None):
    square_height = config.image_array_shape[0] / config.fmap_size[0]
    square_width = config.image_array_shape[1] / config.fmap_size[1]

    x = real_x // square_width
    y = real_y // square_height

    if bbox_upleft!=1:
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

    # Set classifier model to extract features from
    # fmap_model = VGG16(weights='imagenet', include_top=False, input_shape=config.image_array_shape)
    fmap_model = init_panoptic_predictor()

    # Load training data
    path_fp = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_"

    #Iterate for different fovea sizes
    for fovea_size in [100, 75, 50]:

        # Extract from train and validation datasets
        for trainval in ["train.json", "valid.json"]:

            path_fp_train = path_fp + trainval
            fp = open(path_fp_train)
            training_data = json.load(fp)  # list of dictionaries with each experiment
            image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"

            count = 0
            count_files = 0

            # vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

            file_size = 2500  # 3000
            total_size = len(training_data)

            print("TOTAL SIZE: " + str(total_size))

            n_files = int(math.ceil(total_size / file_size))
            out = 0
            obs_in_file = []
            print("NUMBER OF FILES: " + str(n_files))

            # Iterate over experiments
            for i_file in range(n_files):
                initf_time = time.time()

                # Declare rnn inputs and outputs
                rnn_x = []
                rnn_x_nearest = []
                rnn_x_max = []

                label_encodings = []
                rnn_y = []

                start = int(i_file * file_size)
                finish = min(int(i_file * file_size + file_size), len(training_data))
                obs_aux = 0

                for obs in training_data[start:finish]:

                    # Check how many fp it took to detect object
                    for i in range(obs["length"]):
                        if fixated_object(obs["X"][i], obs["Y"][i], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                                          obs["bbox"][3]):
                            break

                    # Ignore observations that took longer than 6 fp to find object
                    if i > config.sequence_len - 1:
                        out += 1
                        print(out)
                        count += 1
                        continue

                    obs_aux += 1
                    # Get image
                    class_name = obs["task"]
                    image_id = obs["name"]
                    image_path = image_dir + "/" + class_name + "/" + image_id
                    img = Image.open(image_path)
                    img_array = img_to_array(img)

                    # holder for features of foveated images
                    #fix = np.empty((0, config.fmap_size[0], config.fmap_size[1], config.fmap_size[2]))
                    fix_nearest = np.empty((0, config.fmap_size[0] * 2, config.fmap_size[1] * 2, config.fmap_size_panoptic[2]))
                    fix_max = np.empty((0, config.fmap_size[0], config.fmap_size[1], config.fmap_size_panoptic[2]))

                    # holder for fp indexes ground truth
                    gt_fixs = np.zeros((0, config.fmap_size[0] * config.fmap_size[1]))

                    # iterate over fp
                    # for i_fp in range(obs["length"]):
                    for i_fp in range(config.sequence_len):
                        true_i_fp = i_fp
                        # if sequence is shorter than 6, repeat last fp
                        if i_fp >= obs["length"]:
                            true_i_fp = obs["length"] - 1
                            x = obs["X"][true_i_fp]
                            y = obs["Y"][true_i_fp]

                        else:

                            x = obs["X"][true_i_fp]
                            y = obs["Y"][true_i_fp]

                            # Foveate image on current fp

                            # fov_image = array_to_img(fov_array)
                            # fov_image.show()

                            # Add fixation features to array
                            #fmap = get_fmap(img_array, x, y, fmap_model, fovea_size, smooth_foveate)
                            fmap_nearest, fmap_max = get_fmap(img_array, x, y, fmap_model, fovea_size, smooth_foveate)

                        #fix = np.append(fix, fmap, axis=0)
                        fix_nearest = np.append(fix_nearest, np.expand_dims(fmap_nearest, axis=0), axis=0)
                        fix_max = np.append(fix_max, np.expand_dims(fmap_max, axis=0), axis=0)


                    for i_fp in range(1, config.sequence_len + 1):
                        true_i_fp = i_fp
                        #For shorter sequences save last fp of sequence until seq len is completed
                        if i_fp > obs["length"] - 1:
                            true_i_fp = obs["length"] - 1

                        # Last fp prediction points to same point
                        elif i_fp == config.sequence_len:
                            true_i_fp = config.sequence_len - 1
                        # Get coord index of next fp
                        next_fix = np.zeros(config.fmap_size[0] * config.fmap_size[1])

                        # next_i = i_fp - 1
                        next_x = obs["X"][true_i_fp]  # [next_i]
                        next_y = obs["Y"][true_i_fp]  # [next_i]
                        grid_next_x, grid_next_y = realcoord2gridcoord(next_x, next_y)
                        idx_next = gridcoord2ind(grid_next_x, grid_next_y)
                        # one hot encode next fixation
                        next_fix[idx_next] = 1

                        next_fix = np.expand_dims(next_fix, axis=0)
                        gt_fixs = np.append(gt_fixs, next_fix, axis=0)

                    #rnn_x.append(fix)
                    rnn_x_nearest.append(fix_nearest)
                    rnn_x_max.append(fix_max)

                    rnn_y.append(gt_fixs)
                    label_encodings.append(one_hot_encoder(class_name))
                    img.close()

                    count += 1
                    print(str(count) + "/" + str(len(training_data)))

                #rnn_x = np.array(rnn_x)
                rnn_x_nearest = np.array(rnn_x_nearest)
                rnn_x_max = np.array(rnn_x_max)
                rnn_y = np.array(rnn_y)
                # rnn_x = np.array(rnn_x, dtype=object)
                # rnn_y = np.array(rnn_y, dtype=object)
                # filename = '/Volumes/DropSave/Tese/dataset/valid_scanpaths_fov100_filtered_' + str(i_file)

                #filename = '/Volumes/DropSave/Tese/dataset/sequences_fixated_in_6_padded_truncated/valid_scanpaths_fov'+\
                #           str(fovea_size)+'_filtered_' + str(i_file)

                filepath = '/Volumes/DropSave/Tese/dataset/sequences_fixated_in_6_padded_truncated/panoptic/'
                filename = 'valid_scanpaths_fov'+str(fovea_size)+'_filtered_' + str(i_file)

                fnearest = filepath + 'nearest/' + filename
                fmax = filepath + 'max/' + filename

                #np.savez_compressed(filename, rnn_x=rnn_x, label_encodings=label_encodings, rnn_y=rnn_y)
                np.savez_compressed(fnearest, rnn_x=rnn_x_nearest, label_encodings=label_encodings, rnn_y=rnn_y)
                np.savez_compressed(fmax, rnn_x=rnn_x_max, label_encodings=label_encodings, rnn_y=rnn_y)

                count_files += 1

                print("***** SAVED FILES " + str(count_files) + "/" + str(n_files) + " *****")
                endf_time = time.time()
                print("Elapsed time for file " + str(count))
                print(timedelta(seconds=endf_time - initf_time))
                obs_in_file.append(obs_aux)

            last_time = time.time()
            print("TOTAL TIME: ")
            print(timedelta(seconds=last_time - first_time))
            local_time = time.ctime(last_time)
            print("Local time:", local_time)
            print(path_fp_train)
            print(obs_in_file)
