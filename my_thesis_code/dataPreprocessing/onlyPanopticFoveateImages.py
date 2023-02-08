import pickle
import os

import laplacian_foveation as fv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import MaxPool2D
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

from glob import glob

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
    return F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0).permute(1, 2, 0).numpy()

def get_panoptic_high_low(predictor, img, blur):
    low = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if not blur: # high res
        out = predictor(np.array(img))["panoptic_seg"]
    else:
        out = predictor(np.array(low))["panoptic_seg"]
    return out

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
    Y, X = np.ogrid[:config.fmap_size_panoptic[0], :config.fmap_size_panoptic[1]]
    x = np.random.randint(low = 0, high = 16, size=(20, 1))
    y = np.random.randint(low = 0, high = 10, size=(20, 1))
    x = np.repeat(y[:, :, np.newaxis], 16, axis=2)
    y = np.repeat(y[:, np.newaxis, :], 10, axis=1)
    Y = torch.from_numpy(Y).unsqueeze(0).repeat((20, 1,1))
    X = torch.from_numpy(X).unsqueeze(0).repeat((20, 1,1))
    dist = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).numpy()
    mask = (dist <= 1).astype(np.float64)

    x_grid = np.array([8])
    x = np.repeat(np.repeat(x_grid[:, np.newaxis, np.newaxis], 16, axis=2), 20, axis=0)


    ''' dir = "/Users/beatrizpaula/Desktop/images_HL_DCBs/hardPanoptic/batchSize256/*.npz"
    for path in glob(dir):
        with np.load(path, allow_pickle=True) as fp:
            prev_rnn_x = fp["rnn_x"]
            label_encodings = fp["label_encodings"]
            rnn_y = fp["rnn_y"]


        rnn_x = []
        for obs in prev_rnn_x:
            d = {}
            for key in ['H', 'L2', 'L5', 'L7']:
                prev_dcb = obs[key]
                print(prev_dcb.shape)
                dcb = np.squeeze(MaxPool2D( pool_size=(2, 2))(np.expand_dims(prev_dcb, axis=0)), axis=0)
                d[key] = dcb
            rnn_x.append(d)

        np.savez_compressed(path, rnn_x=rnn_x, label_encodings=label_encodings, rnn_y=rnn_y)

'''




    '''import torch

    lr_path = "/Users/beatrizpaula/Downloads/000000445411.pth-2.tar"
    hr_path = "/Users/beatrizpaula/Downloads/000000445411.pth.tar"
    lr = torch.load(lr_path)
    hr = torch.load(hr_path)


    h, w, x, y = 20, 32, 10, 16
    r = 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    mask = mask.astype(np.float32)
    final_mask = torch.from_numpy(mask)
    final_mask2 = final_mask.unsqueeze(0).repeat(hr.size(0), 1, 1)

    lr2 = (1 - final_mask2) * lr + final_mask2 * hr

    history_map = torch.zeros((hr.size(-2), hr.size(-1)))
    history_map = (1 - final_mask2[0]) * history_map + final_mask2[0] * 1
'''





    '''first_time = time.time()
    local_time = time.ctime(first_time)
    print("Local time:", local_time)

    # Set classifier model to extract features from
    # fmap_model = VGG16(weights='imagenet', include_top=False, input_shape=config.image_array_shape)
    fmap_model = init_panoptic_predictor()

    # Load training data
    save_path = "/DATA/beatriz_cabarrao/beatriz_paula/dataset/panoptic/high_low"
    categories_path = glob("/DATA/beatriz_cabarrao/beatriz_paula/dataset/resized_images/*")
    #categories_path = glob("/Volumes/DropSave/Tese/dataset/resized_images/*")

    # Get categories dir
    for cat_idx, cat_path in enumerate(categories_path):

        category = cat_path.split("/")[-1]
        save_cat_path = os.path.join(save_path, category)
        if not os.path.exists(save_cat_path):
            os.mkdir(save_cat_path)

        # Iterate over images
        img_paths = glob(cat_path + "/*")
        for img_idx, img_path in enumerate(img_paths):

            img_id = img_path.split("/")[-1].split(".")[0]
            # Iterate for different blurs
            for blur in [0, 2, 5, 7]:
                img = Image.open(img_path).convert('RGB').resize((512, 320))
                if blur:
                    img = img.filter(ImageFilter.GaussianBlur(radius=blur))

                out = fmap_model(np.array(img))["panoptic_seg"]


                if blur:
                    hl = "_L" + str(blur)
                else:
                    hl = "_H"

                save_img_path = os.path.join(save_cat_path, img_id + hl)

                with open(save_img_path, "wb") as fp:
                    pickle.dump(out, fp)

            print(img_idx, "/", len(img_paths), " DONE")

        print("Category ", cat_idx, "/", len(categories_path), " DONE")




'''