import os
import numpy as np
import json
from glob import glob
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import realcoord2gridcoord, gridcoord2ind, gridcoord2realcoord
import config
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange



def bbox2inds(bbox):
    x, y, w, h = bbox
    x_init, y_init = realcoord2gridcoord(x, y)
    x_end, y_end = realcoord2gridcoord(x + w, y + h)

    inds = []

    for j in range(y_init, y_end + 1):
        for i in range(x_init, x_end + 1):
            inds.append(gridcoord2ind(i, j))

    return inds


# returns (up_left, low_right) coord of rectangle expanded to cells
def bbox2cellsbboxcoord(bbox):
    x, y, w, h = bbox
    upper_left_grid = realcoord2gridcoord(x, y, bbox_upleft=1)
    lower_right_grid = realcoord2gridcoord(x + w, y + h)

    cellsbboxcoord = list()
    cellsbboxcoord.append(list(gridcoord2realcoord(upper_left_grid[0], upper_left_grid[1])))
    cellsbboxcoord.append(list(gridcoord2realcoord(lower_right_grid[0], lower_right_grid[1])))

    for i in range(len(cellsbboxcoord[0])):
        cellsbboxcoord[0][i] = cellsbboxcoord[0][i] - 0.5 * config.grid_cell_size[i]
    for i in range(len(cellsbboxcoord[1])):
        cellsbboxcoord[1][i] = cellsbboxcoord[1][i] + 0.5 * config.grid_cell_size[i]

    return cellsbboxcoord


if __name__ == "__main__":
    test_dir = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
    vr = "firstTry"  # <-------- Select model version
    predict_dir = os.path.join("/Volumes/DropSave/Tese/trainedModels", vr, "testing")

    img_directory = "/Volumes/DropSave/Tese/dataset/resized_images/"  # ********

    with open(test_dir) as fp:
        test_dict = json.load(fp)

    # Get all predict file paths:
    predict_paths = glob(predict_dir + "/*/*")

    top = [1, 3, 5]

    when = np.full(len(predict_paths), -1)
    found = np.zeros((len(top), len(predict_paths)))

    task = "keyboard"
    id = "000000217269"
    img_path = img_directory + task + "/" + id + ".jpg"
    # Open image file
    image = Image.open(img_path)
    bbox = test_dict[id][task]

    # font = ImageFont.truetype("arial.ttf", 18)


    # Iterate over predict files
    for exp_idx, path in enumerate(predict_paths):
        test_exp = np.load(path)
        seqs = test_exp["seqs"]

        # Get task and figure id from path
        path_folders = path.split("/")
        name = path_folders[-1].split(".")[0]
        task = path_folders[-2]

        x, y, w, h = bbox = test_dict[name][task]

        x_init, y_init = realcoord2gridcoord(x, y)
        x_end, y_end = realcoord2gridcoord(x + w, y + h)

        inds = bbox2inds(bbox)

        for i, fp in enumerate(seqs[0]):
            if fp in inds:
                when[exp_idx] = i
                break

        for t_pos in range(len(top)):
            for t in range(top[t_pos] + 1):
                for i, fp in enumerate(seqs[t]):
                    if fp in inds:
                        found[t_pos][exp_idx] = 1
                        break
                if(found[t_pos][exp_idx] == 1): break


    accuracy = np.sum(found, axis=1) / len(predict_paths)
    print("Accuracy: ", accuracy)
    bins = range(config.sequence_len)
    plt.hist(when, bins=bins)

    plt.show()

    freqs = np.zeros(config.sequence_len)
    for i in range(len(predict_paths)):
        if when[i] != -1:
            freqs[when[i]] += 1

    print(freqs)

