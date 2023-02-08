import json
import os.path
import random
import math

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.image import resize_with_pad
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
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
from dataPreprocessing.foveateImages import smooth_foveate, ind2gridcoord, gridcoord2realcoord
from evaluation.accuracyCOCO import bbox2inds

from datetime import timedelta
from time import time, ctime

classSizes_path = "/detection/class_sizes.json"
image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"
save_train_data = "/Volumes/DropSave/Tese/dataset/detection_dataset"

def cropImage(fov_arr, x, y):
    vgg_size = [224, 224]
    coco_size = [320, 512]

    x_start, x_end = x - int(0.5*vgg_size[1]), x + int(0.5*vgg_size[1])

    if x_start < 0:

        x_end -= x_start
        x_start -= x_start
    elif x_end >= coco_size[1]:

        shift = x_end - coco_size[1] + 1
        x_end -= shift
        x_start -= shift

    y_start, y_end = y - int(0.5*vgg_size[0]), y + int(0.5*vgg_size[0])


    if y_start < 0:

        y_end -= y_start
        y_start -= y_start
    elif y_end >= coco_size[0]:

        shift = y_end - coco_size[0] + 1
        y_end -= shift
        y_start -= shift
    ret = fov_arr[y_start:y_end, x_start:x_end, :]
    return ret
classes_dict = {
    "bottle": ["beer bottle", "pill bottle", "pop bottle", "water bottle", "wine bottle"],
    "bowl": ["mixing bowl", "soup bowl"],
    "car": ["ambulance", "beach wagon", "cab", "convertible", "jeep", "limousine", "minivan", "Model T", "racer",
            "sports car"],
    "chair": ["throne", "barber chair", "folding chair", "rocking chair"],
    "clock": ["analog clock", "wall clock"],
    "cup": ["cup"],
    "fork": [],
    "knife": ["cleaver", "letter opener"],
    "keyboard": ["computer keyboard"],
    "laptop": ["laptop", "notebook"],
    "microwave": ["microwave"],
    "mouse": ["mouse"],
    "oven": [],
    "potted plant": ["pot", "daisy", "yellow lady's slipper"],
    "sink": [],
    "stop sign": ["street sign"],
    "toilet": ["toilet seat"],
    "tv": ["television"]
}

'''classes_dict = {
            "bottle": [440, 720, 737, 898, 907],
            "bowl": [659, 809],
            "car": [407, 436, 468, 511, 609, 627, 656, 661, 751,
                    817],
            "chair": [857, 423, 559, 765],
            "clock": [409, 892],
            "cup": [969],
            "fork": [],
            "knife": [499, 623],
            "keyboard": [508],
            "laptop": [620, 681],
            "microwave": [651],
            "mouse": [673],
            "oven": [],
            "potted plant": [738, 985, 986],
            "sink": [],
            "stop sign": [919],
            "toilet": [861],
            "tv": [851]
}'''

if __name__ == "__main__":
    #Load pretrained vgg
    classifier = VGG16()


    for fovea in [100, 75, 50]:
        #Grouped classes



        fixations_dir = "/Users/beatrizpaula/Desktop/Tese/my_thesis_code/detection/fineTuneClassifierByClasses"

        model_dir = "/Volumes/DropSave/Tese/trainedModels"
        model_name = "fov100_batch256_normal_rnny_onehot_label"
        model_path = os.path.join(model_dir, model_name)

        image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"

        # dictionary with object bbox info
        test_dataset_dir = test_dir = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
        with open(test_dataset_dir) as fp:
            test_dict = json.load(fp)

        results = dict()
        save_dir = os.path.join(model_path, "detection_preTrained")

        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        classes_aux = list(classes_dict.keys())
        #classes_aux = ["stop sign"]
        #classes_aux.sort(reverse=True)
        flag = 0
        for task_idx, task in enumerate(classes_aux):
            #print(task)
            task_start = time()
            results[task] = dict()
            results[task]["prediction"] = []
            results[task]["gt"] = []

            exp_paths = glob(os.path.join(model_path, "testing", task, "*.npz"))

            for obs_idx, test_path in enumerate(exp_paths):
                flag = 0

                # Load Image
                image_name = test_path.split("/")[-1].split(".")[0] + ".jpg"
                image_path = os.path.join(image_dir, task, image_name)
                img = Image.open(image_path)
                original_img_arr = img_to_array(img)

                image_key = image_name.split(".")[0]
                box = test_dict[image_key][task]
                inds = bbox2inds(box)
                predicted = []
                gt = []

                with np.load(test_path, allow_pickle=True) as f:
                    seq = f["seqs"][0]

                for fp in seq:
                    grid_x, grid_y = ind2gridcoord(fp)
                    x, y = gridcoord2realcoord(grid_x, grid_y)

                    # Foveate and crop image
                    fov_arr = smooth_foveate(original_img_arr, x, y, fovea)
                    img_cp = np.zeros((224, 224, 3))
                    img_cp = np.copy(cropImage(fov_arr, int(x), int(y)))

                    input = preprocess_input(img_cp)
                    input = np.expand_dims(input, axis=0)
                    # Predict classification
                    out = classifier.predict(input)
                    decoded = decode_predictions(out, top=1000)
                    decoded_classes = list(np.transpose(np.array(decoded))[1])
                    if len(classes_dict[task]) > 0:
                        final_pred_probs = 0
                        for equiv in classes_dict[task]:
                            equiv = "_".join(equiv.split())
                            idx_aux = decoded_classes.index(equiv)
                            #idx_aux = equiv
                            final_pred_probs += decoded[0][idx_aux][-1]
                        if final_pred_probs >= decoded[0][0][-1]:
                            final_pred = 1
                            flag = 1
                        else:
                            final_pred = 0
                        predicted.append(final_pred)

                    if fp in inds:
                        gt.append(1)
                    else:
                        gt.append(0)
                    results[task]["prediction"].append(predicted)
                    results[task]["gt"].append(gt)
                str_aux = ""
                if flag: str_aux += "Positive Predict!"
                print(obs_idx, "/", len(exp_paths), str_aux)
                with open(os.path.join(save_dir, "fov" +str(fovea)+"_detection_probabilities_" + task +"_v2.json"), "w") as fp:
                    json.dump(results[task], fp)
                task_end = time()
                print(timedelta(seconds=(task_end - task_start)))

            with open(os.path.join(save_dir, "fov"+str(fovea)+"_detection_probabilities_v2.json"), "w") as fp:
                json.dump(results, fp)






