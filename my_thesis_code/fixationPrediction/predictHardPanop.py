import json
import os
import sys
from datetime import timedelta
from time import time, ctime

import numpy as np
from PIL import Image, ImageFilter
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from glob import glob
from keras import backend as K

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import one_hot_encoder, smooth_foveate, gridcoord2ind, \
    init_panoptic_predictor, get_panoptic_features, get_fmap_hard_panoptic, update_mask, inds2mask
from dataPreprocessing.taskEncodingHeatmap import heat_map_encoder
from evaluation.accuracyCOCO import eval

heatmaps_dir = "/Volumes/DropSave/Tese/dataset/taskencoding_heatmap.npz"
heatmaps = np.load(heatmaps_dir)["label_heatmap"]


# Function that scales fmap coordinates to actual image coordinates
def scale(coord_x, coord_y):
    coord_x = config.image_array_shape[0] / config.cnn_output_shape[0] * (coord_x + 1 / 2)
    coord_y = config.image_array_shape[1] / config.cnn_output_shape[1] * (coord_y + 1 / 2)
    return coord_x, coord_y

def ind2sub(array_shape, ind):
    y = ind // array_shape[1]
    x = ind % array_shape[1]
    return x, y


#ALTERARRRRR
# Mask has already been updated with new fixations
def save_fmaps(fixations, fmaps, hr, lr, mask):
    for idx, fix in enumerate(fixations):
        #key = mask[idx].tobytes()
        key = tuple(mask[idx].flatten())
        if not key in fmaps.keys():
            fmaps[key] = get_fmap_hard_panoptic(hr, lr, mask[idx])

def save_data(seqs, f_maps, label_encoding, vr, img_id, task, test_valid):
    try:
        os.mkdir(os.path.join(vr, test_valid))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(vr, test_valid, task))
    except FileExistsError:
        pass
    np.savez_compressed(os.path.join( vr, test_valid, task, img_id + '.npz'), seqs=seqs, fmaps=f_maps,
                        l_encoding=label_encoding)
    #np.savez_compressed(os.path.join(vr, "testing", task, img_id + '.npz'), seqs=seqs)

def beam_search(model, fmap_model, img_array, task, label_encoding_format, mem_size, vr, img_name,
                foveate_function=smooth_foveate, panoptic=1, hard_panop=None, dcbs=None, test_valid="testing"):
    if hard_panop == None:
        hard_panop = [1,2]

    img_id = img_name + ".jpg"

    hr = dcbs[img_id]['H']
    lr = dcbs[img_id]['L']

    # get task encoding
    task_encoding = one_hot_encoder(task)
    if label_encoding_format:  #heatmap
        task_encoding = heat_map_encoder(label_encoding_format, task_encoding, heatmaps)

    # Setting variable holders
    #   20 Best sequence predictions with ind fp predictions
    seqs = np.zeros((mem_size, config.sequence_len), dtype=int)
    fmaps = {}

    if panoptic:
        fmap_size = config.fmap_size_panoptic
    else:
        fmap_size = config.fmap_size

    task_encoding_1seq = np.transpose(task_encoding[:, np.newaxis, np.newaxis, np.newaxis], (1, 2, 3, 0))
    task_encoding_1seq = np.repeat(np.repeat(task_encoding_1seq, fmap_size[0], axis=1), fmap_size[1], axis=2)

    #   Feature map input to fixation prediction model
    fmap_rnn = np.full((mem_size, config.sequence_len, fmap_size[0], fmap_size[1], fmap_size[2]), -1, dtype="float64") # <-------------------
    #   Set foveation function
    #   Features map extracting model

    # First fixation point is center of fmap
    x_grid, y_grid = fmap_size[1] // 2, fmap_size[0] // 2
    center_ind = gridcoord2ind(x_grid, y_grid)
    seqs[:, 0] = center_ind

    center_inds = [center_ind]
    center_inds = np.repeat(center_ind, mem_size, axis=0)

    mask = inds2mask(center_inds)
    mask_pre = np.zeros((mem_size, fmap_size[0], fmap_size[1]))
    keys = []

    # Get image coordenates of fp
    x_img, y_img = scale(x_grid, y_grid)
    # Get fmap and save fmap with ind as key

    key = tuple(mask[0].flatten())

    fmaps[key] = get_fmap_hard_panoptic(hr, lr, mask[0])


    '''# ******** Visualize foveated Image ************
    fov_array = foveate_function(img_array, x_img, y_img, config.fovea_size)
    fov_image = array_to_img(fov_array)
    #fov_image.show()'''

    # Set rnn input for prediction
    fmap_rnn[:, 0] = fmaps[key]

    # Predict first fp after center
    probs = model.predict(
        [np.expand_dims(np.array(list(fmap_rnn[:, 0])), axis=0), np.repeat(np.expand_dims(task_encoding_1seq, axis=0), mem_size, axis=1)])[:, -1:,:]

    max_probs_inds = np.argpartition(probs.flatten(), -min(probs.size, mem_size))[
                     -min(probs.size, mem_size):]  # select 20 regions with highest probabilities

    seqs[:, 1] = max_probs_inds
    #update mask
    mask_pre = mask.copy()
    mask = inds2mask(max_probs_inds) #mask is of size (20,10,16)
    mask = update_mask(mask, mask_pre)
    save_fmaps(seqs[:, 1], fmaps, hr, lr, mask)

    # Update rnn input with new fp
    for (i, fix) in enumerate(max_probs_inds):
        key = tuple(mask[i].flatten())
        fmap_rnn[i, 1] = fmaps[key]

    # for each prediction of FP use rnn with all previous predictions in sequence
    for current_fix in range(2, config.sequence_len):
        task_encoding = np.repeat(task_encoding_1seq, current_fix, axis=0)
        # Predict next fp
        out_loop = model.predict([fmap_rnn[:, :current_fix], np.repeat([task_encoding], mem_size, axis=0)])
        probs = out_loop[:, -1:, :] # ***************
        # Select 20 highest predictions
        max_probs_inds = np.argpartition(probs.flatten(), -min(probs.size, mem_size))[
                         -min(probs.size, mem_size):]

        # converts max prob indexes to :
        # x->region of image with max_prob as ind,
        # y->which sequence does max_prob belong
        sub = ind2sub((mem_size, probs.shape[2]), max_probs_inds)

        # reorders sequence to maintain max_prob order when adding new FP
        seqs[:, :current_fix] = seqs[sub[1], :current_fix]
        # adds new fixation point to previous sequences
        seqs[:, current_fix] = sub[0]

        # reorder mask_prev
        mask_pre = mask.copy()
        mask_pre[:, :, :] = mask_pre[sub[1], :, :]
        # update mask with new inds
        mask = inds2mask(sub[0])
        mask = update_mask(mask, mask_pre)

        # reorders fmap input to guarantee sequence order
        fmap_rnn[:, :current_fix] = fmap_rnn[sub[1], :current_fix]
        # saves fmaps in new fixations
        save_fmaps(seqs[:, current_fix], fmaps, hr, lr, mask)
        # Updates fmap input
        # sub[0] -> list of inds corresponding to list of max_probs
        for i in range(mem_size):
            key = tuple(mask[i].flatten())
            fmap_rnn[i, current_fix] = fmaps[key]

    # Gets inds of ordered probabilities
    sorted_inds = np.argsort(-probs.flatten()[max_probs_inds])

    # seqs was in max_probs_ins order
    # seqs is now ordered by highest probability of last FP
    seqs = seqs[sorted_inds]

    save_data(seqs, fmaps, task_encoding, vr, img_name, task, test_valid)

    fmaps.clear()
    del fmaps
    del fmap_rnn
    del mask
    del mask_pre
    del out_loop
    del probs
    del max_probs_inds
    del hr
    del lr
    del task_encoding
    del task_encoding_1seq

    return seqs

test_pairs_path = "/Volumes/DropSave/Tese/dataset/test_pairs_id.json"

def predict(vr, path=test_pairs_path, test_valid="testing"):
    # Load testing data

    fp = open(path)
    data = json.load(fp)
    metrics = []

    # Images directory
    image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"

    # Clear tensorflow activity
    keras.backend.clear_session()

    # Set memory size of beam_search
    mem_size = config.mem_size

    # Trained Fixation Prediction model
    # rnn_path = '/Users/beatrizpaula/Downloads/test5Julbatch128_5.h5'
    # rnn_path = "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_onehot_rnny_onehot_label" \
    #           "/fov100_batch256_onehot_rnny_onehot_label.h5 "

    t = dict()

    dirs_w_test = glob("/Volumes/DropSave/Tese/trainedModels/*/testing")
    real_i_dir = 0

    # fmap_model = VGG16(weights='imagenet', include_top=False, input_shape=config.image_array_shape) # <-----------------------------
    fmap_model = init_panoptic_predictor()

    rnn_dirs = vr
    rnn_dirs.sort()
    print(rnn_dirs)
    dirs_len = len(rnn_dirs)
    for i_dir, rnn_path in enumerate(rnn_dirs):
        K.clear_session()
        hard_panop = [0, 0]

        vr = "/".join(rnn_path.split("/")[:-1])
        model_name = vr.split("/")[-1]
        if "hardPanop" in model_name:
            panoptic = 1
            sig = 2
            hard_panop = [1, sig]
            print(hard_panop)
        print(vr)
        panoptic = 1
        sig = 2
        hard_panop = [1, sig]

        flag = 0

        try:
            os.mkdir(os.path.join(vr, test_valid))
        except FileExistsError:
            flag = 1
            if len(glob(os.path.join(vr, test_valid, "*"))) <= 1:
                flag = 0

        if flag: continue

        print("Predicting: ", real_i_dir + 1, "/", dirs_len)

        timedeltas = []

        rnn = load_model(rnn_path)

        # vr = "/Volumes/DropSave/Tese/trainedModels/firstTry"

        print("MODEL: ", vr)

        # check task encoding
        if "heatmap_label2d" in rnn_path:
            label_encoding = 2
        elif "heatmap_label" in rnn_path:
            label_encoding = 1
        else:
            label_encoding = 0
        if "normal_rnny" in rnn_path:
            normal_rnny = 1
        else:
            normal_rnny = 0

        label_encoding = 0
        normal_rnny = 1

        # Fazer iterar sobre test data
        model_start = time()
        local_time = ctime(model_start)
        print("Local time:", local_time)
        start = time()
        count = 0
        dcbs = {}
        for obs in data:

            image_start = time()
            # Load image to array and get task encoding
            image_id = obs["name"]
            class_name = obs["task"]
            image_path = image_dir + "/" + class_name + "/" + image_id
            img = Image.open(image_path).convert('RGB').resize((512, 320))
            if hard_panop[0]:
                img_L = img = img.filter(ImageFilter.GaussianBlur(radius=hard_panop[1]))
            img_array = img_to_array(img)
            # task_encoding = one_hot_encoder(class_name)
            img_name = image_id.split('.')[0]
            if image_id not in dcbs.keys():
                dcbs[image_id] = {'H': get_panoptic_features(fmap_model, np.array(img)),
                                  'L': get_panoptic_features(fmap_model, np.array(img_L))}

            # Use beam search to find sequence with higher probability
            beam_search(rnn, fmap_model, img_array, class_name, label_encoding, mem_size, vr, img_name,
                        foveate_function=smooth_foveate, hard_panop=hard_panop, dcbs=dcbs, test_valid=test_valid)
            time_d = timedelta(seconds=time() - image_start)
            # their_output = seqs[0, 1:]
            # seqs0 = seqs[0]
            count += 1
            print(count, '/', len(data), ' PREDICTED')
            print(time_d)
            timedeltas.append(time_d)
            del img, img_L
        del dcbs

        print(f'test time: {(time() - start)}')
        model_end = time()
        print("TOTAL TIME: ")
        print(timedelta(seconds=model_end - model_start))
        local_time = ctime(model_end)
        print("Local time:", local_time)
        print("AVERAGES: ", sum(timedeltas, timedelta(0)) / len(timedeltas))
        t[vr] = timedeltas

        save_path = vr + "/testTimes.txt"
        with open(save_path, 'w') as file:
            file.write("TOTAL: \n" + str(timedelta(seconds=model_end - model_start)) + "\n")
            file.write("PARTIALS: \n: " + str(timedeltas) + "\n")

            file.close()

        del rnn
        real_i_dir += 1
        print("***** PREDICTED: ", real_i_dir, "/", dirs_len, " *****")
        metrics.append(eval(test_valid))

    np.savez_compressed('times16.npz', times=t)

    for model in metrics:
        for m in model:
            print(m)
        print()

if __name__ == "__main__":
    vr = [
        "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_onehot_rnny_onehot_label_hardPanop_rpanop1_sig2_labelConcat_rnnyDamp_dense_f40_drop05_conv3_f20_l1_w01_tdf_dense_f160_soft/fov100_batch256_onehot_rnny_onehot_label_hardPanop_rpanop1_sig2_labelConcat_rnnyDamp_dense_f40_drop05_conv3_f20_l1_w01_tdf_dense_f160_soft.h5",
        "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_onehot_rnny_onehot_label_hardPanop_rpanop1_sig2_labelConcat_rnnyDamp_dense_f40_conv3_f20_tdf_dense_f160_soft/fov100_batch256_onehot_rnny_onehot_label_hardPanop_rpanop1_sig2_labelConcat_rnnyDamp_dense_f40_conv3_f20_tdf_dense_f160_soft.h5"]

    t_v = [["test", "testing"], ["valid", "validating"]]
    select = 0
    vr = glob("/Volumes/DropSave/Tese/trainedModels/*vgg*/*.h5")
    path = "/Volumes/DropSave/Tese/dataset/" + t_v[select][0] + "_pairs_id.json"
    predict(vr, path=path, test_valid=t_v[select][1])




