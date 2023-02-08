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

def eval(test_valid):
    if test_valid == "testing":
        aux = "test"
        pos = -1
    else:
        aux = "valid"
        pos = -2

    test_dir = "/Volumes/DropSave/Tese/dataset/" + aux + "_dictionary.json"
    # vr = "firstTry"  # <-------- Select model version
    top = [1, 3, 5]
    # predict_dir = os.path.join("/Volumes/DropSave/Tese/trainedModels", vr, "testing")
    predict_dirs = glob("/Volumes/DropSave/Tese/trainedModels/*/" + test_valid)
    predict_dirs = ["/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label/testing"]
    predict_dirs = glob("/Volumes/DropSave/Tese/trainedModels/hardPanop*/testing")
    print(predict_dirs)
    predict_dirs= ["/Volumes/DropSave/Tese/trainedModels/panop_feat_vgg_arch/testing"]

    img_directory = "/Volumes/DropSave/Tese/dataset/resized_images/"  # ********

    with open(test_dir) as fp:
        test_dict = json.load(fp)

    # iterate over tested models:
    predict_dirs.sort()
    all = []
    for predict_dir in predict_dirs:

        '''if not "hardPanop" in predict_dir:
            print("not")
            continue'''

        # check if accuracy has been calculated
        eval_dir = "/".join(predict_dir.split("/")[:-1])
        if test_valid == "validating":
            eval_dir = os.path.join(eval_dir, test_valid)
        save_dir = os.path.join(eval_dir, "eval")
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
            #continue

        # Get all predict file paths:
        predict_paths = glob(predict_dir + "/*/*.npz")

        when = np.full(len(predict_paths), -1)
        found = np.zeros((len(top), len(predict_paths)))

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

            # Check when object is fixated for top1 prediction
            for i, fp in enumerate(seqs[0]):
                if fp in inds:
                    when[exp_idx] = i
                    break

            # Iterate over top
            for t_pos in range(len(top)):
                # iterate over predictions in top
                for t in range(top[t_pos]):
                    # iterate over fp in prediction
                    for i, fp in enumerate(seqs[t]):
                        if fp in inds:  # if fp detected object
                            found[t_pos][exp_idx] = 1
                            break
                    if found[t_pos][exp_idx] == 1:
                        break  # if object has been found there is no need to keep checking top sequences

        this_name = eval_dir.split("/")[pos]
        print(this_name)
        accuracy = np.sum(found, axis=1) / len(predict_paths)
        print("Accuracy: ", accuracy)

        # Check frequency of object detection throughout sequence len
        freqs = np.zeros(config.sequence_len)
        for i in range(len(predict_paths)):
            if when[i] != -1:
                freqs[when[i]] += 1

        print(freqs)

        cumulative = []

        for i in range(config.sequence_len):
            sum_freq = 0
            sum_freq = sum(freqs[:(i + 1)])
            cumulative.append(sum_freq)

        print(cumulative)
        cumulative = np.array(cumulative)
        cumulative = cumulative / len(predict_paths)

        fixations_to_detection_axis = list(range(config.sequence_len))
        plt.plot(fixations_to_detection_axis, cumulative, marker='|')
        plt.title('Cumulative probability of fixating the target', fontsize=14)
        plt.xlabel('Number of Fixations Made to Target', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.ylim([0, 1])
        # plt.grid(True)

        plt.savefig(save_dir + "/cumulative.png")
        plt.clf()

        area = 0
        for i in range(1, config.sequence_len):
            rectangle_area = cumulative[i - 1]  # h, w is always 1
            triangle_area = (cumulative[i] - cumulative[i - 1]) * 0.5
            area += (rectangle_area + triangle_area)

        print(area)

        this = [this_name, accuracy, freqs, cumulative, area]
        all.append(this)

        save_path = save_dir + "/evaluationMetrics.json"

        d = {"top": top, "accuracies": accuracy.tolist(), "when": freqs.tolist(), "cumulative": cumulative.tolist(),
             "area": area.tolist()}

        with open(save_path, 'w') as file:
            json.dump(d, file)

        save_path = save_dir + "/evaluationMetrics.txt"

        with open(save_path, 'w') as file:
            file.write("top: " + str(top) + "\n")
            file.write("accuracies: " + str(accuracy) + "\n")
            file.write("when: " + str(freqs) + "\n")
            file.write("cumulative: " + str(cumulative) + "\n")
            file.write("cumulative area: " + str(area) + "\n")
            file.close()

    return all




if __name__ == "__main__":
    #, "validating"

    for test_valid in ["testing"]:
        eval(test_valid)

    '''paths = glob("/Volumes/DropSave/Tese/trainedModels/*/*histories.npz")

    for path in paths:
        with np.load(path, allow_pickle=True) as fp:
            m = fp["histories"].item()

        print(path.split("/")[-2])
        print(m["val_categorical_accuracy"], "\n")'''



