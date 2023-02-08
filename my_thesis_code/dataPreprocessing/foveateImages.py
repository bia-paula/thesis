import laplacian_foveation as fv
import numpy as np
from PIL import Image, ImageDraw

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import math
import time

'''import torch
import torch.nn.functional as F'''
#from detectron2 import model_zoo
'''from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog'''


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
def inds2mask(inds, r = 1):
    #seq_len = x.shape[0]
    x, y = ind2gridcoord(inds) #arrays of size mem_size
    Y, X = np.ogrid[:config.fmap_size_panoptic[0], :config.fmap_size_panoptic[1]]
    Y = np.expand_dims(Y, axis=0)
    X = np.expand_dims(X, axis=0)
    Y = np.repeat(Y, len(y), axis=0)
    X = np.repeat(X, len(x), axis=0)
    x, y = x[:, np.newaxis, np.newaxis], y[:, np.newaxis, np.newaxis]
    x = np.repeat(x, config.fmap_size_panoptic[1], axis=2)
    y = np.repeat(y, config.fmap_size_panoptic[0], axis=1)
    #Y = np.repeat(np.expand_dims(Y, axis=2), seq_len, axis=2)
    #X = np.repeat(np.expand_dims(X, axis=2), seq_len, axis=2)
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r # array of shape: (mem_size, 10, 16)
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
    mask = np.transpose(mask.unsqueeze(0).repeat(config.fmap_size_panoptic[-1], 1, 1, 1), (1,2,0,3))
    hr = np.repeat(np.expand_dims(hr, axis=3), config.sequence_len, axis=3)
    lr = np.repeat(np.expand_dims(lr, axis=3), config.sequence_len, axis=3)
    if accumulate:
        for idx in range(1, len(ind)):
            mask[:, :, :, idx] += mask[:, :, :, idx-1]
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
    #f_map_pixel = get_panoptic_features(features_model, fov_image)

    return f_map
    #return f_map_pixel



def realcoord2gridcoord(real_x, real_y, bbox_upleft=1):
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
    # Load training data
    #path_fp_train = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
    # path_fp_train = "/Users/beatrizpaula/Downloads/human_scanpaths_TP_trainval_valid.json"
    path_fp = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_" # <---------------------------------------------
    # image_dir =  "/Users/beatrizpaula/Desktop/resized_images"

    # Set classifier model to extract features from
    fmap_model = VGG16(weights='imagenet', include_top=False, input_shape=config.image_array_shape)
    #fmap_model = init_panoptic_predictor()

    save_filepath = '/Volumes/DropSave/Tese/dataset/sequences_by_nfixations/' # <--------------------------------------------------


    # Iterate for different fovea sizes
    for fovea_size in [50]:

        # Extract from train and validation datasets
        for trainval in ["train.json", "valid.json"]:

            path_fp_train = path_fp + trainval
            fp = open(path_fp_train)
            training_data = json.load(fp)  # list of dictionaries with each experiment
            image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"
            name_trainval = trainval.split(".")[0]

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

                if trainval == "train.json" and i_file <= 5:
                    continue

                initf_time = time.time()

                # Declare rnn inputs and outputs
                data = dict()

                for idx in range(7):
                    data[idx] = {"rnn_x": [], "label_encodings": [], "rnn_y": []}

                start = int(i_file * file_size)
                finish = min(int(i_file * file_size + file_size), len(training_data))
                obs_aux = 0

                '''# Declare rnn inputs and outputs
                rnn_x = []
                label_encodings = []
                rnn_y = []
                to_file = 0'''

                for idx_obs, obs in enumerate(training_data[start:finish]):
                    obs_start = time.time()

                    flag = 0

                    # Check how many fp it took to detect object
                    for i_fixated in range(min(obs["length"], 7)):
                        if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                                          obs["bbox"][3]):
                            flag=1
                            break

                    # Ignore observations that took longer than 6 fp to find object
                    if not flag:
                        out += 1
                        print(out)
                        count += 1
                        continue

                    # Get image
                    class_name = obs["task"]
                    image_id = obs["name"]
                    image_path = image_dir + "/" + class_name + "/" + image_id
                    img = Image.open(image_path)
                    img_array = img_to_array(img)

                    # holder for features of foveated images
                    fix = np.empty((0, config.fmap_size[0], config.fmap_size[1], config.fmap_size[2]))
                    #fix_nearest = np.empty((0, config.fmap_size[0], config.fmap_size[1], config.fmap_size_panoptic[2]))
                    #fix_max = np.empty((0, config.fmap_size[0], config.fmap_size[1], config.fmap_size_panoptic[2]))

                    # holder for fp indexes ground truth
                    gt_fixs = np.zeros((0, config.fmap_size[0] * config.fmap_size[1]))

                    # iterate over fp
                    for i_fp in range(7):
                    #for i_fp in range(min(obs["length"], 7)):
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

                            fmap = get_fmap(img_array, x, y, fmap_model, fovea_size, smooth_foveate) # <----------------------------- check
                            #fix = np.append(fix, fmap, axis=0)

                            #fmap_nearest, fmap_max = get_fmap(img_array, x, y, fmap_model, config.fovea_size, smooth_foveate)
                        fix = np.append(fix, fmap, axis=0)
                        #fix_nearest = np.append(fix_nearest, np.expand_dims(fmap_nearest, axis=0), axis=0)
                        #fix_max = np.append(fix_max, np.expand_dims(fmap_max, axis=0), axis=0)


                    #for i_fp in range(1, 7 + 1):
                    for i_fp in range(1, min(obs["length"], 7) + 1):
                        true_i_fp = i_fp
                        #For shorter sequences save last fp of sequence until seq len is completed
                        if i_fp > obs["length"] - 1:
                            true_i_fp = obs["length"] - 1

                        # Last fp prediction points to same point
                        #elif i_fp == config.sequence_len:
                            #true_i_fp = config.sequence_len - 1
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

                    data[min(obs["length"], 7)-1]["rnn_x"].append(fix)
                    #rnn_x.append(fix)
                    #rnn_x_nearest.append(fix_nearest)
                    #rnn_x_max.append(fix_max)

                    #rnn_y.append(gt_fixs)
                    #label_encodings.append(one_hot_encoder(class_name))
                    data[min(obs["length"], 7)-1]["rnn_y"].append(gt_fixs)
                    data[min(obs["length"], 7)-1]["label_encodings"].append(one_hot_encoder(class_name))
                    img.close()

                    count += 1
                    #to_file += 1
                    obs_end = time.time()
                    print(str(count) + "/" + str(len(training_data)), " -> (", str(obs["length"]), ") ", timedelta(seconds=obs_end - obs_start))

                    '''if to_file == file_size:
    
                        rnn_x = np.array(rnn_x)
                        label_encodings = np.array(label_encodings)
                        rnn_y = np.array(rnn_y)
    
                        filename = name_trainval + '_scanpaths_fov' + str(fovea_size) + '_panopticIRL_' + str(count_files)
    
                        # fnearest = filepath + 'nearest/' + filename
                        # fmax = filepath + 'max/' + filename
    
                        np.savez_compressed(save_filepath + filename, rnn_x=rnn_x, label_encodings=label_encodings, rnn_y=rnn_y)
                        
    
                        count_files += 1
                        
                        # reset variables
                        rnn_x = []
                        label_encodings = []
                        rnn_y = []
                        to_file = 0
    
                if to_file > 0:
                    rnn_x = np.array(rnn_x)
                    label_encodings = np.array(label_encodings)
                    rnn_y = np.array(rnn_y)
    
                    
                    filename = name_trainval + '_scanpaths_fov' + str(fovea_size) + '_panopticIRL_' + str(count_files)
    
                    # fnearest = filepath + 'nearest/' + filename
                    # fmax = filepath + 'max/' + filename
    
                    np.savez_compressed(save_filepath + filename, rnn_x=rnn_x, label_encodings=label_encodings, rnn_y=rnn_y)
    
                    count_files += 1
    
                    print("Saved last file smaller:", to_file)'''



                for idx in range(7):
                    data[idx]["rnn_x"] =  np.array(data[idx]["rnn_x"])
                    data[idx]["label_encodings"] = np.array(data[idx]["label_encodings"])
                    data[idx]["rnn_y"] = np.array(data[idx]["rnn_y"])


                #rnn_x = np.array(rnn_x)
                #rnn_x_nearest = np.array(rnn_x_nearest)
                #rnn_x_max = np.array(rnn_x_max)
                #rnn_y = np.array(rnn_y)
                # rnn_x = np.array(rnn_x, dtype=object)
                # rnn_y = np.array(rnn_y, dtype=object)
                # filename = '/Volumes/DropSave/Tese/dataset/valid_scanpaths_fov100_filtered_' + str(i_file)

                #filename = '/Volumes/DropSave/Tese/dataset/sequences_fixated_in_6_padded_truncated/valid_scanpaths_fov'+\
                #           str(config.fovea_size)+'_filtered_' + str(i_file)

                for idx in range(7):

                    filepath = '/Volumes/DropSave/Tese/dataset/sequences_by_nfixations/'
                    name_trainval = trainval.split(".")[0]
                    filename = name_trainval + '_scanpaths_fov'+str(fovea_size)+'_filtered_length' + str(idx) + "_" + str(i_file)

                    #fnearest = filepath + 'nearest/' + filename
                    #fmax = filepath + 'max/' + filename

                    np.savez_compressed(filepath + filename, rnn_x=data[idx]["rnn_x"], label_encodings=data[idx]["label_encodings"], rnn_y=data[idx]["rnn_y"])
                    #np.savez_compressed(fnearest, rnn_x=rnn_x_nearest, label_encodings=label_encodings, rnn_y=rnn_y)
                    #np.savez_compressed(fmax, rnn_x=rnn_x_max, label_encodings=label_encodings, rnn_y=rnn_y)

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

