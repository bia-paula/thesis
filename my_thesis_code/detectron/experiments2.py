import json
import math
import pickle
import random

import matplotlib.pyplot as plt

'''
import torch
import torch.nn.functional as F
import numpy as np'''

from PIL import Image, ImageFilter
'''import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2'''
import numpy as np
import sys
import scipy
'''from tensorflow.keras.models import load_model
from glob import glob
import os
import tensorflow.keras.backend as K
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont'''

'''sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import fixated_object
'''
# from  skimage.measure import block_reduce

'''sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import ind2gridcoord, gridcoord2ind, get_circular_hard_foveate_dcb, inds2mask, \
    get_fmap_hard_panoptic, update_mask, gridcoord2realcoord, realcoord2gridcoord

from fixationPrediction.predictHardPanop import save_fmaps, ind2sub'''

'''from tensorflow.keras.layers import MaxPool3D, Input, Dense, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.models import Model'''

'''from datetime import timedelta

from dataPreprocessing.foveateImages import smooth_foveate'''

def func(rnn_y, sigma):
    x, y = ind2gridcoord(np.argmax(rnn_y))
    pos = [y, x]
    pos = np.expand_dims(np.expand_dims(np.array(pos), axis=1).repeat(10, axis=1), axis=2).repeat(16, axis=2)
    m = (scipy.mgrid[0: 10, 0: 16] - pos) ** 2
    sum = m[0] + m[1]
    e = np.exp(- sum / (2 * sigma ** 2))
    return (e / (2 * sigma * np.pi)).flatten()

def d(x, y):
    a = (x[1]-x[0])**2
    b = (y[1]-y[0])**2
    return math.sqrt(a + b)


from math import pi, exp
if __name__ == '__main__':
    A, B, C, D = [6,2,4,7]
    count = 0
    for h1 in [A, B, C, D]:
        left = [A, B, C, D]
        left.remove(h1)
        for h2 in left:
            left2 = left.copy()
            left2.remove(h2)
            for s1 in left2:
                left3 = left2.copy()
                left3.remove(s1)
                for s2 in left3:
                    if h1 * 10 + h2 < 24 and s1 * 10 + s2 < 60:
                        count+=1

    # Check duplicates:
    check = set([A, B, C, D])
    if len(check) == 4:
        den = 1
    elif len(check) == 3:
        den = 2
    elif len(check) == 2:
        val = A
        flag = 0
        for el in [A, B, C, D]:
            if val == el:
                flag+=1
        if flag == 2:
            den = 2 * 2 # two pairs
        else:
            den = 3 * 2 # one triple
    else:
        den = 4*3*2

    decreasing = [0] * len(blocks)
    for i in range(1, len(blocks)):
        if blocks[i] <= blocks[i - 1]
            decreasing[i] = decreasing[i-1]+1

    increasing = [0] * len(blocks)
    for i in range(len(blocks)-2, -1, -1):
        if blocks[i] <= blocks[i + 1]
            increasing[i] = increasing[i + 1] + 1


        '''path = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
    
        with open(path) as fp:
            data = json.load(fp)
    
        double = []
        tasks = dict()
        for c in config.classes:
            tasks[c] = set()
        for image in data.keys():
            if len(data[image].keys()) > 1:
                double.append(image)
                for t1 in data[image].keys():
                    for t2 in data[image].keys():
                        if t1 != t2:
                            tasks[t1].add(t2)
    
        path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
    
        with open(path) as fp:
            train = json.load(fp)
    
        w = {}
        h = {}
        a = {}
    
        for d in [w, h, a]:
            for cat in config.classes:
                d[cat] = []
    
        for obs in train:
            cat = obs["task"]
            _, _, wb, hb = obs['bbox']
            w[cat].append(wb)
            h[cat].append(hb)
            a[cat].append(wb * hb)
    
        W, H, A = [[], [], []], [[], [], []], [[], [], []]
    
        for d, D in [[w, W], [h, H], [a, A]]:
            for task in w.keys():
                D[0].append(np.median(d[task]))
                D[1].append(np.mean(d[task]))
                D[2].append(np.std(d[task]))
    
        for dist in [W, H, A]:
            for m in range(3):
                print(dist[m])
    
        hours = set()
        minutes = set()
        values = [A, B, C, D]
        '''












    '''path = "/Volumes/DropSave/Tese/trainedModels/dual/*dualLSTM*"

    trained_models = glob(path)
    predicted_models = []
    to_predict = []

    for model in trained_models:
        if os.path.isdir(os.path.join(model, "testing")):
            predicted_models.append(model.split("/")[-1])
        else:
            to_predict.append(model.split("/")[-1])

    path = "/Volumes/DropSave/Tese/trainedModels/dual/fov*"

    paths = glob(path)
    paths.sort()

    header = ["r", "cumul", "gt", "task", "depth", "activation"]

    metrics = []
    for i in [1, 3, 5]:
        metrics.append("accuracy-" + str(i))
    metrics.append("area")
    for i in range(7):
        metrics.append("tfp-" + str(i))

    header.extend(metrics)

    f = []
    f.append(header)

    data_dict = dict()

    for idx_model, model in enumerate(paths):
        vr = model.split("/")[-1]
        _, _, r, gt, _, task, _, _, d, _, act = vr.split("_")
        if "accum" in r:
            accum = 1
            r = r.split("a")[0].split("r")[-1]
        else:
            accum = 0
            r = r.split("r")[-1]

        json_list = glob(os.path.join(model, "eval", "*.json"))
        json_path = json_list[0]
        print(idx_model, "/", len(paths), model)

        with open(json_path) as fp:
            data = json.load(fp)

        accs = data["accuracies"]
        area = data["area"]
        cumul = data["cumulative"]

        line = [r, accum, gt, task, d, act]
        line.extend(accs)
        line.append(area)
        line.extend(cumul)

        f.append(line)

        key = str([int(r), accum, int(d), act])[1:-1]
        data_dict[key] = accs[0]

    save_dir = "/Users/beatrizpaula/Desktop/FP2_all_metrics"
    with open(save_dir, "w") as fp:
        for l in f:
            fp.write(str(l) + "\n")

    save_dir = "/Users/beatrizpaula/Desktop/FP2_all_metrics.json"
    with open(save_dir, "w") as fp:
        json.dump(data_dict, fp)'''

    '''path = "/Volumes/DropSave/Tese/trainedModels/dual/*dualLSTM*"

    trained_models = glob(path)
    predicted_models = []
    to_predict = []

    for model in trained_models:
        if os.path.isdir(os.path.join(model, "testing")):
            predicted_models.append(model.split("/")[-1])
        else:
            to_predict.append(model.split("/")[-1])

    path = "/Volumes/DropSave/Tese/trainedModels/hardPanop*"
    "hardPanop_sig2_r3_normal_rnny_onehot_label_redDepth_1_act_softm"

    paths = glob(path)
    paths.sort()

    header = ["r", "cumul", "gt", "task", "depth", "activation"]

    metrics = []
    for i in [1, 3, 5]:
        metrics.append("accuracy-"+str(i))
    metrics.append("area")
    for i in range(7):
        metrics.append("tfp-"+str(i))

    header.extend(metrics)

    f = []
    f.append(header)

    data_dict = dict()

    for idx_model, model in enumerate(paths):
        vr = model.split("/")[-1]
        _, _, r, gt, _, task, _, _, d, _, act = vr.split("_")
        if "accum" in r:
            accum = 1
            r = r.split("a")[0].split("r")[-1]
        else:
            accum = 0
            r = r.split("r")[-1]

        json_list = glob(os.path.join(model, "eval", "*.json"))
        json_path = json_list[0]
        print(idx_model, "/", len(paths), model)

        with open(json_path) as fp:
            data = json.load(fp)

        accs = data["accuracies"]
        area = data["area"]
        cumul = data["cumulative"]


        line = [r, accum, gt, task, d, act]
        line.extend(accs)
        line.append(area)
        line.extend(cumul)

        f.append(line)

        key = str([int(r), accum, int(d), act])[1:-1]
        data_dict[key] = accs[0]

    save_dir = "/Users/beatrizpaula/Desktop/FP2_all_metrics"
    with open(save_dir, "w") as fp:
        for l in f:
            fp.write(str(l) + "\n")

    save_dir = "/Users/beatrizpaula/Desktop/FP2_all_metrics.json"
    with open(save_dir, "w") as fp:
        json.dump(data_dict, fp)'''


    '''
    print(len(rand_dict))





   with open("/Volumes/DropSave/Tese/dataset/human_scanpaths_random.json") as fp:
        rand_dict = json.load(fp)

    save_dir = "/Volumes/DropSave/Tese/trainedModels"

    try:
        os.mkdir(os.path.join(save_dir, "random", "testing"))
    except FileExistsError:
        pass

    test_data = json.load(open("/Volumes/DropSave/Tese/dataset/test_pairs_id.json"))

    for obs in test_data:

        try:
            os.mkdir(os.path.join(save_dir, "random", "testing", obs["task"]))
        except FileExistsError:
            pass
        seqs = np.zeros((20, 7))
        idxs = []
        for m in range(20):
            idx = random.randrange(len(rand_dict[obs["task"]]))
            idxs.append(idx)
            xs, ys = rand_dict[obs["task"]][idx]
            inds = np.zeros(7)
            for t in range(min(len(xs), 7)):
                x, y = xs[t], ys[t]
                x, y = np.array(x), np.array(y)
                x, y = realcoord2gridcoord(x, y)
                inds[t] = gridcoord2ind(x, y)
            aux = min(len(xs), 7) - 1
            seq = np.full(7, inds[aux])
            seq[:aux+1] = inds[:aux+1]
            seqs[m] = seq

        print(idxs)
        print(seqs, "\n")

        np.savez_compressed(os.path.join(save_dir, "random", "testing", obs["task"], obs["name"].split(".")[0] + '.npz'), seqs=seqs)'''







    '''p = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
    with open(p) as fp:
        data = json.load(fp)

    rand_dict = dict()
    for c in config.classes:
        rand_dict[c] = []

    for obs in data:
        flag = 0
        for i_fixated in range(min(obs["length"], 7)):
            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                flag = 1
                break
        if not flag: continue

        rand_dict[obs["task"]].append([obs["X"], obs["Y"]])
    with open("/Volumes/DropSave/Tese/dataset/human_scanpaths_random.json", "w") as fp:
        json.dump(rand_dict, fp)
    
    
    p = "/Users/beatrizpaula/Desktop/Desktop - Beatriz’s MacBook Air/images_HL_DCBs/high_low/chair/000000578793_L7"

    data = np.load(p, allow_pickle=True)

    print(float(5e-5))

    # creating a image object
    im1 = Image.open("/Volumes/DropSave/Tese/dataset/resized_images/bottle/000000024823.jpg")

    # applying the Gaussian Blur filter
    im2 = im1.filter(ImageFilter.GaussianBlur(radius=4))

    im2.show()

    im3 = Image.fromarray(smooth_foveate(np.array(im1), 256, 160, 50))

    im3.show()

    im4 = np.array(im2)

    inds = [[7, 3], [6,4], [7,4], [8,4], [7,5]]

    all_grid = []

    for ind in inds:
        x, y = ind
        im4[y*32:(y*32+32), x*32:(x*32+32), :] = np.array(im1)[y*32:(y*32+32), x*32:(x*32+32), :]

    im4 = Image.fromarray(im4)
    im4.show()'''










    '''sigma = 2
    matrice = np.zeros((config.fmap_size[0] ,config.fmap_size[1]))  # (10, 16) (y, x)
    for x in range(config.fmap_size[1]):
        for y in range(config.fmap_size[0]):
            d = ((x - 8) ** 2) + ((y - 5) ** 2)
            matrice[y, x] = (1 / (2 * pi * sigma)) * exp(-d / (2 * (sigma ** 2)))

    for task in config.classes:
        path = "/Volumes/DropSave/Tese/trainedModels/classifier/fov100_vgg16_"+task+".h5"

        model = load_model(path)

        print(task, K.eval(model.optimizer.lr))

    path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"

    with open(path) as fp:
        train = json.load(fp)

    W = {}
    H = {}

    for d in [W, H]:
        for cat in config.classes:
            d[cat] = []

    for obs in train:
        cat = obs["task"]
        _, _, w, h = obs['bbox']
        W[cat].append(w)
        H[cat].append(h)'''

    '''w1, w2 = min(W), max(W)
    h1, h2 = min(H), max(H)

    wq1, wq2, wq3 = np.quantile(W,0.25), np.quantile(W,0.5), np.quantile(W,0.75)
    hq1, hq2, hq3 = np.quantile(H, 0.25), np.quantile(H, 0.5), np.quantile(H, 0.75)

    meanW = np.mean(W)
    meanH = np.mean(H)'''

    '''bpW = plt.boxplot(W)
    key = "boxes"
    print(f'{key}: {[item.get_ydata() for item in bpW[key]]}\n')
    plt.show()

    bpH = plt.boxplot(H)
    key = "boxes"
    print(f'{key}: {[item.get_ydata() for item in bpH[key]]}\n')
    plt.show()'''

    '''fig, axs = plt.subplots(1, len(W.keys()), figsize=(50,10))

    width_medians = []
    for i, ax in enumerate(axs.flat):
        ax.boxplot(W[config.classes[i]], showfliers=False)
        ax.set_title(config.classes[i], fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0, 350)
        width_medians.append(np.quantile(W[config.classes[i]],0.5))

    #plt.tight_layout()
    plt.savefig("classes_widths.png")
    plt.show()

    print(min(width_medians))
    print(max(width_medians))

    fig, axs = plt.subplots(1, len(H.keys()), figsize=(50, 10))
    height_medians = []
    for i, ax in enumerate(axs.flat):
        ax.boxplot(H[config.classes[i]], showfliers=False)
        ax.set_title(config.classes[i], fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0, 250)
        height_medians.append(np.quantile(H[config.classes[i]],0.5))

    # plt.tight_layout()
    plt.savefig("classes_heights.png")
    plt.show()

    print(min(height_medians))
    print(max(height_medians))

    d = {"height": height_medians, "width": width_medians}

    with open("class_sizes.json", "w") as fp:
        json.dump(d, fp)'''


    ### Ordenar modelos por test accuracies
    '''metrics_path = glob("/Volumes/DropSave/Tese/trainedModels/*hardPanop*/eval/evaluationMetrics.json")
    metrics_path.sort()

    all_models_acc = []
    all_models_area = []
    all_models_name = []
    cat_acc = []
    val_loss = []
    last_loss = []
    loss = []

    # Test metrics
    for path in metrics_path:

        with open(path) as fp:
            m = json.load(fp)

        name = path.split("/")[-3]
        all_models_acc.append(m["accuracies"][0])
        all_models_area.append(m["area"])
        all_models_name.append(name)

        l_path = path.split("/")[:-2]
        l_path.append(name + "_histories.npz")
        train_path = "/".join(l_path)

        model_path = path.split("/")[:-2]
        model_path.append(name + ".h5")
        model_path = "/".join(model_path)

        with np.load(train_path, allow_pickle=True) as f:
            d = f["histories"].item()

            cat_acc.append(d["val_categorical_accuracy"][-1])
            val_loss.append(d["val_loss"][-1])
            last_loss.append(d["loss"][-1])

        m = load_model(model_path)
        loss.append(m.loss)


    ranked = np.flip(np.argsort(all_models_acc)).tolist()

    sorted_acc = [all_models_acc[i] for i in ranked]
    sorted_area = [all_models_area[i] for i in ranked]
    sorted_name = [all_models_name[i] for i in ranked]
    sorted_val_acc = [cat_acc[i] for i in ranked]
    sorted_val_loss = [val_loss[i] for i in ranked]
    sorted_last_loss = [last_loss[i] for i in ranked]
    sorted_loss = [loss[i] for i in ranked]

    config_dir = "/Volumes/DropSave/Tese/trainedModels/"

    sorted_cfg = []

    for model in all_models_name:
        m = load_model(os.path.join(config_dir, model, model + ".h5"))
        sorted_cfg.append(m.get_config())

    del all_models_acc, all_models_name, all_models_area, cat_acc, val_loss, last_loss
'''




    # print(c)

    '''# Get test dcbs:
    for obs, id in [[obs_short, idx2], [obs_long, idx]]:
        x0, y0 = np.array([16, 10]) // 2
        ind = np.full(7, gridcoord2ind(x0, y0))
        ind[1:] = np.argmax(rnn_y[id], axis=1)[:-1]
        hr = dicts_list[id]['H']
        l_key = "L" + str(2)
        lr = dicts_list[id][l_key]
        rnn_x.append(get_circular_hard_foveate_dcb(ind, hr, lr, r=1, accumulate=1))
        inds.append(ind)'''

    '''

    myfunc_vec = np.vectorize(get_circular_hard_foveate_dcb)

    Y, X = np.ogrid[:10, :16]

    Y = np.repeat(np.expand_dims(Y, axis=2), 7, axis=2)
    X =np.repeat( np.expand_dims(X, axis=2), 7, axis=2)

    d = np.sqrt((X - 8) ** 2 + (Y - 5) ** 2)
    



    hard_panoptic = [0, 2]
    rnn_x = []
    for idx, obs in enumerate(dicts_list):
        x0, y0 = np.array([16, 10]) // 2
        ind = np.full(7, gridcoord2ind(x0, y0))
        ind[1:] = np.argmax(rnn_y[idx], axis=1)[:-1]
        hr = block_reduce(obs['H'], (2,2,1), np.max)
        l_key = "L" + str(hard_panoptic[1])
        lr = block_reduce(obs[l_key], (2,2,1), np.max)

        px, py = ind2gridcoord(ind)

        rnn_x.append(get_circular_hard_foveate_dcb(ind, hr, lr, accumulate=0))
    rnn_x = np.array(rnn_x)'''

    '''
if __name__ == '__main__':
    # Load pretrained panoptic_fpn
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

    img_path = "test1.png"

    im = Image.open(img_path).convert('RGB').resize((512, 320))


    out = predictor(np.array(im))["panoptic_seg"]
    out2 = []
    out2.append(out)
    out2.append(out)


    with open("pickletest.pickle", "wb") as file:
        pickle.dump(out2, file)

    seg, info = predictor(np.array(im))["panoptic_seg"]

    feat = torch.zeros([80 + 54, 320, 512])
    for pred in info:
        mask_pre = (seg == pred['id'])
        mask = (seg == pred['id']).float()
        print(pred['id'])
        print(mask)
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat[pred['category_id'] + 80, :, :] = mask

    last1 = F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0).numpy()

    last2 = F.max_pool2d(feat, (32, 32)).numpy()

    featnp = feat.numpy()'''

'''v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))


    window_name = 'image'
    draw_fov = v.draw_panoptic_seg(seg, info)
    #cv2.imshow(window_name, draw_fov.get_image()[:, :, ::-1])
    #cv2.waitKey()
    #cv2.destroyAllWindows()



    img_path = "test2.png"

    im_fov = Image.open(img_path).convert('RGB').resize((512, 320))

    open_cv_image = np.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    out_fov = predictor(open_cv_image)
    seg_fov, info_fov = predictor(open_cv_image)["panoptic_seg"]

    feat_fov = torch.zeros([80 + 54, 320, 512])
    for pred in info_fov:
        mask_pre = (seg_fov == pred['id'])
        mask = (seg_fov == pred['id']).float()
        print(pred['id'])
        print(mask)
        if pred['isthing']:
            feat_fov[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat_fov[pred['category_id'] + 80, :, :] = mask

    final_int1 = feat_fov.unsqueeze(0)
    final_int2 = F.interpolate(feat_fov.unsqueeze(0), size=[20, 32])
    final = F.interpolate(feat_fov.unsqueeze(0), size=[20, 32]).squeeze(0)




    #v = Visualizer(im_fov, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    #window_name = 'image'
    #draw_fov = v.draw_panoptic_seg(seg_fov, info_fov)
    #cv2.imshow(window_name, draw_fov.get_image()[:, :, ::-1])
    #cv2.waitKey()
    #cv2.destroyAllWindows()'''
