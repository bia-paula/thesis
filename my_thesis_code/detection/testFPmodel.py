import os
from glob import glob
import numpy as np
import json
from datetime import timedelta
from time import time, ctime

from fineTuneClassifiers import cropImage, classSizes_path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.image import resize_with_pad
from tensorflow.keras.applications.vgg16 import preprocess_input

from PIL import Image
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import smooth_foveate, ind2gridcoord, gridcoord2realcoord
from evaluation.accuracyCOCO import bbox2inds

if __name__ == "__main__":
    model_dir = "/Volumes/DropSave/Tese/trainedModels"


    model_name = "fov100_batch256_normal_rnny_onehot_label"

    classifier_dir = "/Volumes/DropSave/Tese/trainedModels/classifier"
    image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"

    model_path = os.path.join(model_dir, model_name)

    # dictionary with object bbox info
    test_dataset_dir = test_dir = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
    with open(test_dataset_dir) as fp:
        test_dict = json.load(fp)

    with open(classSizes_path) as fp:
        sizes = json.load(fp)

    heights = sizes["height"]
    widths = sizes["width"]
    for fovea in [75, 50, 100]:
        results = dict()
        save_dir = os.path.join(model_path, "detection")

        for task_idx, task in enumerate(config.classes):
            print(task)
            task_start = time()
            results[task] = dict()
            results[task]["prediction"] = []
            results[task]["gt"] = []

            classifier_name = "fov"+str(fovea)+"_vgg16_" + task + ".h5"
            classifier = load_model(os.path.join(classifier_dir, classifier_name))

            exp_paths = glob(os.path.join(model_path, "testing", task, "*.npz"))

            h, w = heights[task_idx], widths[task_idx]

            for obs_idx, test_path in enumerate(exp_paths):

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
                    input_img = cropImage(fov_arr, x, y, task, heights, widths)
                    img = resize_with_pad(input_img, int(h * 2), int(w * 2))
                    input = preprocess_input(img)
                    input = np.expand_dims(input, axis=0)

                    # Predict classification
                    out = classifier.predict(input)
                    predicted.append(out.tolist()[0][0])
                    if fp in inds:
                        gt.append(1)
                    else:
                        gt.append(0)

                    results[task]["prediction"].append(predicted)
                    results[task]["gt"].append(gt)

                print(obs_idx, "/", len(exp_paths))

            with open(os.path.join(save_dir, "fov"+str(fovea)+"_detection_probabilities_"+task+".json"), "w") as fp:
                json.dump(results[task], fp)
            task_end = time()
            print(timedelta(seconds=(task_end - task_start)))


        with open(os.path.join(save_dir, "detection_probabilities.json"), "w") as fp:
            json.dump(results, fp)
