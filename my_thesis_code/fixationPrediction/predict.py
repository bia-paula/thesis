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
from dataPreprocessing.foveateImages import one_hot_encoder, smooth_foveate, get_fmap, gridcoord2ind, ind2gridcoord, \
    init_panoptic_predictor, get_panoptic_features, gridcoord2mask
from dataPreprocessing.taskEncodingHeatmap import heat_map_encoder

heatmaps_dir = "/Volumes/DropSave/Tese/dataset/taskencoding_heatmap.npz"
heatmaps = np.load(heatmaps_dir)["label_heatmap"]

prob_weights = [0.01, 0.44, 0.36, 0.12, 0.04, 0.02, 0.01]

# Function that scales fmap coordinates to actual image coordinates
def scale(coord_x, coord_y):
    coord_x = config.image_array_shape[0] / config.cnn_output_shape[0] * (coord_x + 1 / 2)
    coord_y = config.image_array_shape[1] / config.cnn_output_shape[1] * (coord_y + 1 / 2)
    return coord_x, coord_y

def ind2sub(array_shape, ind):
    y = ind // array_shape[1]
    x = ind % array_shape[1]
    return x, y

def save_fmaps(fixations, fmaps, im, cnn, fovea_size, f_function):
    for fix in fixations:
        if fix not in fmaps.keys():
            x, y = ind2gridcoord(fix)
            x, y = scale(x, y) #coordenates in original image size
            fmaps[fix] = get_fmap(im, x, y, cnn, fovea_size, f_function)

def save_data(seqs, f_maps, label_encoding, vr, img_id, task):
    try:
        os.mkdir(os.path.join(vr, "testing"))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(vr, "testing", task))
    except FileExistsError:
        pass
    np.savez_compressed(os.path.join( vr, "testing", task, img_id + '.npz'), seqs=seqs, fmaps=f_maps,
                        l_encoding=label_encoding)
    #np.savez_compressed(os.path.join(vr, "testing", task, img_id + '.npz'), seqs=seqs)

def beam_search(model, fmap_model, img_array, task, label_encoding_format, mem_size, vr, img_name,
                foveate_function=smooth_foveate, panoptic=0, hs=0, model_last=None):

    if not hs:
        model_last = model
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
    #   Feature map input to fixation prediction model
    fmap_rnn = np.full((mem_size, config.sequence_len, fmap_size[0], fmap_size[1], fmap_size[2]), -1) # <-------------------
    #   Set foveation function
    #   Features map extracting model

    # First fixation point is center of fmap
    x_grid, y_grid = fmap_size[1] // 2, fmap_size[0] // 2
    center_ind = gridcoord2ind(x_grid, y_grid)
    seqs[:, 0] = center_ind

    # Get image coordenates of fp
    x_img, y_img = scale(x_grid, y_grid)
    # Get fmap and save fmap with ind as key
    fmaps[center_ind] = get_fmap(img_array, x_img, y_img, fmap_model, config.fovea_size, foveate_function)

    # ******** Visualize foveated Image ************
    fov_array = foveate_function(img_array, x_img, y_img, config.fovea_size)
    fov_image = array_to_img(fov_array)
    #fov_image.show()

    # Set rnn input for prediction
    fmap_rnn[:, 0] = fmaps[center_ind]

    # Predict first fp after center
    probs_first = model.predict(
        [np.expand_dims(np.array(list(fmap_rnn[:, 0])), axis=0), np.expand_dims(task_encoding, axis=0)])
    probs = probs_first[:, :1,:]  # ***************

    max_probs_inds = np.argpartition(probs.flatten(), -min(probs.size, mem_size))[
                     -min(probs.size, mem_size):]  # select 20 regions with highest probabilities

    seqs[:, 1] = max_probs_inds
    save_fmaps(seqs[:, 1], fmaps, img_array, fmap_model, config.fovea_size, foveate_function)

    # Update rnn input with new fp
    for (i, fix) in enumerate(max_probs_inds):
        fmap_rnn[i, 1] = fmaps[fix]

    # for each prediction of FP use rnn with all previous predictions in sequence
    for current_fix in range(2, config.sequence_len):
        if current_fix > 2:
            model = model_last
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

        # reorders fmap input to guarantee sequence order
        fmap_rnn[:, :current_fix] = fmap_rnn[sub[1], :current_fix]
        # saves fmaps in new fixations
        save_fmaps(seqs[:, current_fix], fmaps, img_array, fmap_model, config.fovea_size, foveate_function)
        # Updates fmap input
        for (i, fix) in enumerate(sub[0]):
            fmap_rnn[i, current_fix] = fmaps[fix]

    # Gets inds of ordered probabilities
    sorted_inds = np.argsort(-probs.flatten()[max_probs_inds])
    #sorted_inds = np.argsort(-f_probs)

    # seqs was in max_probs_ins order
    # seqs is now ordered by highest probability of last FP
    seqs = seqs[sorted_inds]

    save_data(seqs, fmaps, task_encoding, vr, img_name, task)

    aux = probs.flatten()[max_probs_inds]

    return seqs

if __name__ == "__main__":

    # Load testing data
    path = "/Volumes/DropSave/Tese/dataset/test_pairs_id.json"
    fp = open(path)
    data = json.load(fp)

    # Images directory
    image_dir = "/Volumes/DropSave/Tese/dataset/resized_images"

    # Clear tensorflow activity
    keras.backend.clear_session()

    # Set memory size of beam_search
    mem_size = config.mem_size

    # Trained Fixation Prediction model
    #rnn_path = '/Users/beatrizpaula/Downloads/test5Julbatch128_5.h5'
    #rnn_path = "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_onehot_rnny_onehot_label" \
    #           "/fov100_batch256_onehot_rnny_onehot_label.h5 "

    rnn_dirs = glob("/Volumes/DropSave/Tese/trainedModels/*/*.h5")
    rnn_dirs = ["/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label/fov100_batch256_normal_rnny_onehot_label.h5"]

    rnn_dirs.sort()
    t = dict()
    dirs_len = len(rnn_dirs)

    dirs_w_test = glob("/Volumes/DropSave/Tese/trainedModels/*/testing")
    dirs_wo = set(rnn_dirs) - set(dirs_w_test)
    dirs_len = len(dirs_wo)
    real_i_dir = 0

    fmap_model = VGG16(weights='imagenet', include_top=False, input_shape=config.image_array_shape) # <-----------------------------
    #fmap_model = init_panoptic_predictor()

    for i_dir, rnn_path in enumerate(rnn_dirs):
        K.clear_session()
        vr = "/".join(rnn_path.split("/")[:-1])
        model_name = vr.split("/")[-1]
        if "hardPanop" in model_name:
            panoptic = 1
            sig = int(model_name.split("sig")[-1])
            hard_panop = [1, sig]
        print(vr)
        flag = 0

        try:
            os.mkdir(os.path.join(vr, "testing"))
        except FileExistsError:
            if len(glob(os.path.join(vr, "testing", "*"))) == 0:
                pass
            else:
                continue

        print("Predicting: ", real_i_dir + 1, "/", dirs_len)


        timedeltas = []

        rnn = load_model(rnn_path)
        print(rnn.summary())

        #vr = "/Volumes/DropSave/Tese/trainedModels/firstTry"

        print("MODEL: ", vr)

        #check task encoding
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

        # Fazer iterar sobre test data
        model_start = time()
        local_time = ctime(model_start)
        print("Local time:", local_time)
        start = time()
        count = 0
        dcbs={}
        for obs in data:

            image_start = time()
            # Load image to array and get task encoding
            image_id = obs["name"]
            class_name = obs["task"]
            image_path = image_dir + "/" + class_name + "/" + image_id
            img = Image.open(image_path).convert('RGB').resize((512, 320))
            img_array = img_to_array(img)
            #task_encoding = one_hot_encoder(class_name)
            img_name = image_id.split('.')[0]


            # Use beam search to find sequence with higher probability
            beam_search(rnn, fmap_model, img_array, class_name, label_encoding, mem_size, vr, img_name,
                        foveate_function=smooth_foveate, panoptic=0, hs=0)
            time_d = timedelta(seconds=time() - image_start)
            #their_output = seqs[0, 1:]
            #seqs0 = seqs[0]
            count += 1
            print(count, '/', len(data), ' PREDICTED')
            print(time_d)
            timedeltas.append(time_d)

        print(f'test time: {(time() - start)}')
        model_end = time()
        print("TOTAL TIME: ")
        print(timedelta(seconds=model_end - model_start))
        local_time = ctime(model_end)
        print("Local time:", local_time)
        print("AVERAGES: ", sum(timedeltas, timedelta(0))/len(timedeltas))
        t[vr]=timedeltas

        save_path = vr + "/testTimes.txt"
        with open(save_path, 'w') as file:
            file.write("TOTAL: \n" + str(timedelta(seconds=model_end - model_start)) + "\n")
            file.write("PARTIALS: \n: " + str(timedeltas) + "\n")

            file.close()

        del rnn
        print("***** PREDICTED: ", i_dir + 1, "/", dirs_len, " *****")
        real_i_dir += 1



    np.savez_compressed('times16.npz', times=t)



