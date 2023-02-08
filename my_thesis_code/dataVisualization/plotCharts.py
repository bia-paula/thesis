import matplotlib.pyplot as plt
import json

import numpy as np
import math

from glob import glob
import os
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import fixated_object, gridcoord2realcoord, realcoord2gridcoord, ind2gridcoord, \
    gridcoord2ind
from evaluation.accuracyCOCO import bbox2inds

save_dir = "/Users/beatrizpaula/Desktop/Tese/images"

def getImagesPerClass():
    dir = "/Volumes/DropSave/Tese/dataset/resized_images"
    values = []
    for task in config.classes:
        values.append(len(glob(os.path.join(dir, task, "*"))))
    m = np.mean(values)
    values.append(m)
    return values

def plotBarChart(x_axis, y_axis):
    fig = plt.figure()
    color_p = plt.get_cmap('tab20').colors
    colors=[]
    for i in range(len(x_axis)):
        colors.append(color_p[i])
    colors[-1] = 'black'

    plt.bar(x_axis, y_axis, color=colors)
    plt.xticks(rotation=90, fontsize=12)
    plt.xlabel("Target classes", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "imagesPerClass"))
    plt.show()

def getWhenFixatedTrainingDatset():
    path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"

    with open(path) as fp:
        training_data = json.load(fp)

    when = np.zeros((19,7))
    all = []
    comul = np.zeros((19,7))
    comul2 = np.zeros((7))
    for task in config.classes:
        all.append(0)
        print(task)

    for idx_obs, obs in enumerate(training_data):

        flag = 0
        # Get image
        class_name = obs["task"]
        all[config.classes.index(class_name)]+=1
        for i_fixated in range(min(obs["length"], 7)):
            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                flag = 1
                when[config.classes.index(class_name)][i_fixated] += 1
                comul2[i_fixated] += 1
                break

            if not flag: continue

    for task in config.classes:
        for i in range(1, 7):
            when[config.classes.index(task), i] += when[config.classes.index(task), i-1]
        #sum = np.sum(when[config.classes.index(task)])
        print(task)
        print(all[config.classes.index(task)])
        comul[config.classes.index(task)] = when[config.classes.index(task)]/all[config.classes.index(task)]

        when[-1] += (when[config.classes.index(task)] / 18)
        comul[-1] += (comul[config.classes.index(task)] / 18)
    comul[-1] = comul[-1]
    print(comul[-1, -1])

    for i in range(1, 7):
        comul2[i] += comul2[i-1]
    total = np.sum(all)
    comul2 = comul2/total

    return when, comul, comul2


def getCumulTFPTestModel(path_predicted, dual=0):
    test_dataset_dir = ["/Volumes/DropSave/Tese/dataset/test_dictionary.json",
                        "/Volumes/DropSave/Tese/dataset/test_dictionary_TA.json"]
    with open(test_dataset_dir[0]) as fp:
        test_dict = json.load(fp)
    with open(test_dataset_dir[1]) as fp:
        test_dict2 = json.load(fp)

    for k in test_dict2.keys():
        if k not in test_dict:
            test_dict[k] = test_dict2[k]
        else:
            for task in test_dict2[k].keys():
                test_dict[k][task] = test_dict2[k][task]

    all_cumulative = []
    print("Dictionary made")
    when_dual = []
    freqs_dual = np.zeros(config.sequence_len)
    total = 0

    for task_idx, task_path in enumerate(glob(os.path.join(path_predicted, "*"))):
        images_task_path = glob(os.path.join(task_path, "*"))
        total_task = 0

        when = np.full(len(images_task_path), -1)
        when_dual.append(np.full(len(images_task_path), -1))
        for exp_idx, image_path in enumerate(images_task_path):

            with np.load(image_path) as test_exp:
                seqs = test_exp["seqs"]

            # Get task and figure id from path
            path_folders = image_path.split("/")
            name = path_folders[-1].split(".")[0]
            task = path_folders[-2]

            if len(test_dict[name][task]) != 0:
                bbox = test_dict[name][task]
                total_task += 1
                total += 1
            else:
                continue

            inds = bbox2inds(bbox)

            for i, fp in enumerate(seqs[0]):
                if fp in inds:
                    when[exp_idx] = i
                    when_dual[task_idx][exp_idx] = i
                    break

        #print(task, when_dual[task_idx])
        # Check frequency of object detection throughout sequence len
        freqs = np.zeros(config.sequence_len)
        for i in range(len(images_task_path)):
            if when[i] != -1:
                freqs[when[i]] += 1
                freqs_dual[when[i]] += 1
            #print(task, when[6] / total_task)
        print(task, freqs_dual, total_task)

        cumulative = []

        for i in range(config.sequence_len):
            sum_freq = 0
            sum_freq = sum(freqs[:(i + 1)])
            cumulative.append(sum_freq)

        cumulative = np.array(cumulative)
        cumulative = cumulative / total_task

        all_cumulative.append(cumulative)
    if dual:
        cumulative = []
        all_cumulative = []
        for i in range(config.sequence_len):
            sum_freq = 0
            sum_freq = sum(freqs_dual[:(i + 1)])
            cumulative.append(sum_freq)

        cumulative = np.array(cumulative)
        cumulative = cumulative / total
        print(total)
        print(cumulative)
        for i in range(len(config.classes)):
            all_cumulative.append(cumulative)

    return np.array(all_cumulative)

def getWhenFixatedTestDatset():
    path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_test.json"

    with open(path) as fp:
        training_data = json.load(fp)

    when = np.zeros((19,21))
    all = []
    comul = np.zeros((19,21))
    comul2 = np.zeros((21))
    for task in config.classes:
        all.append(0)
        print(task)
    lens = []
    missed = {}

    for idx_obs, obs in enumerate(training_data):

        flag = 0
        # Get image
        class_name = obs["task"]
        all[config.classes.index(class_name)]+=1
        lens.append(obs["length"])

        for i_fixated in range(min(obs["length"], 21)):

            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                flag = 1
                when[config.classes.index(class_name)][i_fixated] += 1
                comul2[i_fixated] += 1
                break

            if not flag: continue

        if not flag:
            if not obs["name"] in missed.keys():
                missed[obs["name"]] = dict()
            if not obs["task"] in missed[obs["name"]].keys():
                missed[obs["name"]][obs["task"]] = 1
            else:
                missed[obs["name"]][obs["task"]] += 1
    for task in config.classes:
        for i in range(1, 21):
            when[config.classes.index(task), i] += when[config.classes.index(task), i-1]
        #sum = np.sum(when[config.classes.index(task)])
        print(task)
        print(all[config.classes.index(task)])
        comul[config.classes.index(task)] = when[config.classes.index(task)]/all[config.classes.index(task)]

        when[-1] += (when[config.classes.index(task)] / 18)
        comul[-1] += (comul[config.classes.index(task)] / 18)
    comul[-1] = comul[-1]
    print(comul[-1, -1])

    for i in range(1, 21):
        comul2[i] += comul2[i-1]
    total = np.sum(all)
    comul2 = comul2/total

    return when, comul, comul2, lens, missed

def plotLineChart(x_axis, y_axis, labels, colors, xlabel, ylabel):
    plt.figure()
    ax = plt.subplot()
    for t in range(len(x_axis)):
        w = 2
        if t == 18:
            w = 3
        color = colors[t]
        x = x_axis[t]
        y = y_axis[t]
        label = labels[t]
        ax.plot(x, y, label=label, color=color, linewidth=w)

    #plt.legend()
    plt.xlim(0, 6)
    plt.ylim(0, 1)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "datasetCumulProb"))
    plt.show()

def plot_VGG_boxplots(metrics, keys):
    for k in metrics.keys():
        metrics[k] = np.array(metrics[k])

    fovea_acc = {"50": [], "75": [], "100": []}
    for idx, k in enumerate(keys):
        fovea_acc[k[0]].append(metrics["top-1"][idx])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(fovea_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(fovea_acc.keys())
    plt.grid(axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "VGG_accuracy_fovea"))
    fig.show()

    task_acc = {"one-hot": [], "heatmap-1d": [], "heatmap-2d": []}
    for idx, k in enumerate(keys):
        task_acc[k[1]].append(metrics["top-1"][idx])

    print("Heat-map-1d")
    print(np.quantile(task_acc["heatmap-1d"], 0.25))
    print("Heat-map-2d")
    print(np.quantile(task_acc["heatmap-2d"], 1))

    m_2d = np.max(task_acc["heatmap-2d"])
    count = 0
    count_equal = 0
    for aux_idx, aux in enumerate(task_acc["heatmap-1d"]):
        if aux > m_2d:
            count += 1
            count_equal += 1
        elif aux == m_2d:
            count_equal += 1
    print(count/len(task_acc["heatmap-1d"]))
    print(count_equal/len(task_acc["heatmap-1d"]))

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(task_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(["One-Hot", "Heatmap-1D", "Heatmap-2D"])
    plt.grid(axis="y")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "VGG_accuracy_task"))
    fig.show()

    gt_acc = {"one-hot": [], "gaussian": []}
    for idx, k in enumerate(keys):
        gt_acc[k[2]].append(metrics["top-1"][idx])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(gt_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(["One-Hot", "Gaussian"])
    plt.grid(axis="y")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "VGG_accuracy_gt"))
    fig.show()

    batch_acc = {"32": [], "64": [], "128": [], "256": []}
    for idx, k in enumerate(keys):
        batch_acc[k[3]].append(metrics["top-1"][idx])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(batch_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(batch_acc.keys())
    plt.grid(axis="y")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "VGG_accuracy_batch"))
    fig.show()

def grid_dist(x, y):
    a = (x[1] - x[0]) ** 2
    b = (y[1] - y[0]) ** 2
    return math.sqrt(a + b)

def plot_bars_distance_length():
    lens = []
    #lens_model = []
    lens_model = [[], [], [], [], []]
    dist_model = []
    median_dist_model = [[], [], [], [], []]
    std_dist_model = [[], [], [], [], []]

    ps = ["/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing/*",
         "/Users/beatrizpaula/Downloads/fov100_batch32_onehot_rnny_onehot_label/testing/*"]
    ps = ["/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing/*",
         "/Volumes/DropSave/Tese/trainedModels/hardPanop_sig2_r1accum_normal_rnny_onehot_label_redDepth_1_act_sigm/testing/*"]
    ps = ["/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing/*",
          "/Users/beatrizpaula/Downloads/fov100_batch32_onehot_rnny_onehot_label/testing/*",
          "/Volumes/DropSave/Tese/trainedModels/hardPanop_sig2_r1accum_normal_rnny_onehot_label_redDepth_1_act_sigm/testing/*",
        "/Volumes/DropSave/Tese/trainedModels/dual/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_90/testing/*",
        "/Volumes/DropSave/Tese/trainedModels/dual/Concat/fov75_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_90/testing/*"]
    for i_p, p in enumerate(ps):
        tasks = glob(p)
        dist_model.append(dict())
        for i in range(1, 7):
            dist_model[i_p][i] = []
        for task in tasks:
            for obs in glob(os.path.join(task, "*")):
                with np.load(obs) as data:
                    seq = data['seqs'][0]

                for i in range(1, 7):
                    x0, y0 = ind2gridcoord(seq[i - 1])
                    x0, y0 = gridcoord2realcoord(x0, y0)
                    x1, y1 = ind2gridcoord(seq[i])
                    x1, y1 = gridcoord2realcoord(x1, y1)

                    dist_model[i_p][i].append(grid_dist([x0, x1], [y0, y1]))

                l = 0
                for i_fp in reversed(range(1, 7)):
                    if seq[i_fp] != seq[i_fp - 1]:
                        l = i_fp
                        break
                lens_model[i_p].append(l)


        for k in dist_model[i_p].keys():
            print("key:", k)
            median_dist_model[i_p].append(np.mean(dist_model[i_p][k]))
            std_dist_model[i_p].append(np.std(dist_model[i_p][k]))

    print(len(median_dist_model[0]))
    print(len(median_dist_model[1]))
    print(len(median_dist_model[2]))
    print(len(median_dist_model[3]))
    print(len(median_dist_model[4]))
    print(len(std_dist_model[0]))
    print(len(std_dist_model[1]))
    print(len(std_dist_model[2]))
    print(len(std_dist_model[3]))
    print(len(std_dist_model[4]))

    path_fp_train = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
    with open(path_fp_train) as fp:
        training_data = json.load(fp)

    dist_data = []
    for i in range(6):
        dist_data.append([])

    for obs in training_data:
        flag = 0
        for i_fixated in range(min(obs["length"], 7)):
            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                flag = 1
                break
        if not flag:
            continue
        lens.append(min(obs["length"], 7))

        x, y = realcoord2gridcoord(obs["X"][-1], obs["Y"][-1])
        fill = np.full((7,), gridcoord2ind(x, y))

        for i in range(min(obs["length"], 7)):
            x, y = realcoord2gridcoord(obs["X"][i], obs["Y"][i])
            fill[i] = gridcoord2ind(x, y)

        for i in range(1, 7):
            x0, y0 = ind2gridcoord(fill[i - 1])
            x0, y0 = gridcoord2realcoord(x0, y0)
            x1, y1 = ind2gridcoord(fill[i])
            x1, y1 = gridcoord2realcoord(x1, y1)

            dist_data[i - 1].append(grid_dist([x0, x1], [y0, y1]))

    median_dist_data = []
    std_dist_data = []
    for i in range(6):
        median_dist_data.append(np.mean(dist_data[i]))
        std_dist_data.append(np.std(dist_data[i]))

    keys = []
    for idx in range(1, 7):
        k = "{}-{}".format(idx - 1, idx)
        keys.append(k)

    # plt.plot(med[0])
    # Set position of bar on X axis
    barWidth = 0.15
    x = np.arange(len(median_dist_data))
    br1 = [y - 2.5 * barWidth for y in x]
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    plt.bar(br1, median_dist_data, width=barWidth)
    plt.bar(br2, median_dist_model[0], width=barWidth)
    plt.bar(br3, median_dist_model[1], width=barWidth)
    plt.bar(br4, median_dist_model[2], width=barWidth)
    plt.bar(br5, median_dist_model[3], width=barWidth)
    plt.bar(br6, median_dist_model[4], width=barWidth)

    plt.xticks(range(len(median_dist_model[0])), keys, fontsize=12)

    plt.legend(['Training data', 'High-level Features - Gaussian GT', 'High-level Features - One-hot GT',
    'Panoptic Features', "Dual - Architecture A", "Dual - Architecture C"])
    plt.xlabel("Fixation Points", fontsize=14)
    plt.ylabel("Mean distance [pixels]", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ALL_dist_fp"))
    plt.show()

    '''labels, counts = np.unique(lens, return_counts=True)
    total = np.sum(counts)
    counts = counts / total
    plt.bar(labels, counts, align='center', width=1.0)
    plt.gca().set_xticks(labels)
    plt.xticks(fontsize=12)
    plt.ylabel("Frequency as fraction of total sequences", fontsize=14)
    plt.xlabel("Sequence length", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Dual_seq_len_hist"))
    plt.show()'''


def std(arr):
    N = len(arr)
    miu = np.mean(arr)

    num = 0
    for i in range(N):
        num += (num-miu)**2
    s = math.sqrt(num/(N-1))

    return s/math.sqrt(N)

def get_TFP_VGG():
    _, human_all, _ = getWhenFixatedTrainingDatset()

    human = np.mean(human_all, axis=0)
    human_err = np.std(human_all, axis=0)

    beatriz_path = "/Users/beatrizpaula/Downloads/fov100_batch32_onehot_rnny_onehot_label/testing"
    meu_path = "/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing"

    beatriz_all = getCumulTFPTestModel(beatriz_path)
    beatriz = np.mean(beatriz_all, axis=0)
    beatriz_err = np.std(beatriz_all, axis=0)
    beatriz_err = []
    for t in range(7):
        beatriz_err.append(std(beatriz_all[:, t]))
    beatriz_err = np.array(beatriz_err)

    meu_all = getCumulTFPTestModel(meu_path)
    meu = np.mean(meu_all, axis=0)
    meu_err = []
    for t in range(7):
        meu_err.append(std(meu_all[:, t]))
    meu_err = np.array(meu_err)

    rand_path = "/Volumes/DropSave/Tese/trainedModels/random/testing"
    rand_all = getCumulTFPTestModel(rand_path)
    rand = np.mean(rand_all, axis=0)
    rand_err = np.std(rand_all, axis=0)
    rand_err = []
    for t in range(7):
        rand_err.append(std(rand_all[:, t]))
    rand_err = np.array(rand_err)

    x = range(7)

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.errorbar(x, human, yerr=human_err, label="Human", ls="--")
    plt.errorbar(x, meu, yerr=meu_err, label="ConvLSTM - Gaussian GT", )
    plt.errorbar(x, beatriz, yerr=beatriz_err, label="ConvLSTM - One-hot GT")
    plt.errorbar(x, rand, yerr=rand_err, label="Random scanpath")
    plt.xlim([0, 6])
    plt.ylim([0, 1])
    plt.xlabel("Time step")
    plt.ylabel("Target fixation cumulative probability")
    box = ax.get_position()
    '''ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])'''

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", ncol=2, mode="expand")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "VGG_TFP"))
    plt.show()

def scanpath_ratio_func(seq, bbox):
    x, y, w, h = bbox
    x_targ, y_targ = x + w/2, y + h/2
    x_targ, y_targ = realcoord2gridcoord(x_targ, y_targ)
    x_c, y_c = ind2gridcoord(seq[0])
    target_dist = math.sqrt((x_c-x_targ)**2 + (y_c - y_targ)**2)

    scan_dist = 0
    for i in range(1, len(seq)):
        x_0, y_0 = ind2gridcoord(seq[i-1])
        x_1, y_1 = ind2gridcoord(seq[i])
        scan_dist += math.sqrt((x_1-x_0)**2 + (y_1 - y_0)**2)
        x_real, y_real = gridcoord2realcoord(x_1, y_1)


    if scan_dist == 0:
        ratio = 1
    else:
        ratio = target_dist/scan_dist
    return ratio

def get_AUC(cumul_mean):
    area = 0
    for i in range(1, len(cumul_mean)):
        a = 0.5 * (cumul_mean[i-1] + cumul_mean[i])
        area += a
    return area

def plot_Panop_boxplots(data):

    radius_acc = {1: [], 2:[], 3:[]}
    accum_acc = {0: [], 1:[]}
    depth_acc = {1: [], 3: [], 5: []}
    activ_acc = {"sigm": [], "softm": []}

    for k in data.keys():
        k0 = k
        k = k.split(", ")
        r = int(k[0])
        acum = int(k[1])
        d = int(k[2])
        act = k[3][1:-1]

        radius_acc[r].append(data[k0])
        accum_acc[acum].append(data[k0])
        depth_acc[d].append(data[k0])
        activ_acc[act].append(data[k0])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(radius_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(radius_acc.keys())
    plt.grid(axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Panop_accuracy_radius"))
    fig.show()

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(accum_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(["Non-cumulative", "Cumulative"])
    plt.grid(axis="y")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Panop_accuracy_cumulative"))
    fig.show()

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(depth_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(depth_acc.keys())
    plt.grid(axis="y")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Panop_accuracy_depth"))
    fig.show()

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(activ_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(["Sigmoid", "Softmax"])
    plt.grid(axis="y")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Panop_accuracy_activation"))
    fig.show()

def plot_dual_boxplots(metrics, keys):
    for k in metrics.keys():
        metrics[k] = np.array(metrics[k])

    m = "fix_acc"
    label = "Search Accuracy"

    arch_acc = {"fixFirst": [], "detectFirst": [], "concat": []}
    for idx, k in enumerate(keys):
        arch_acc[k[0]].append(metrics[m][idx])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(arch_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(["A","B","C"])
    plt.grid(axis="y")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([0.3, 0.8])
    plt.ylabel(label, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Dual_accuracy_arch_"+m))
    fig.show()

    fovea_acc = {"50": [], "75": [], "100": []}
    for idx, k in enumerate(keys):
        fovea_acc[k[1]].append(metrics[m][idx])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(fovea_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(fovea_acc.keys())
    plt.grid(axis="y")
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim([0.3, 0.8])
    plt.ylabel(label, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Dual_accuracy_fovea_"+m))
    fig.show()

    weight_acc = {"10": [], "25": [], "50": [], "75": [], "90": []}
    for idx, k in enumerate(keys):
        weight_acc[k[2]].append(metrics[m][idx])

    fig, ax = plt.subplots()
    meanprop = dict(marker='v', markeredgecolor='black',
                    markerfacecolor='#2F75B5')
    bp = ax.boxplot(weight_acc.values(), meanprops=meanprop, showmeans=True, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#BDD7EE80")
    for median in bp['medians']:
        median.set_color('#0070C0')
    ax.set_xticklabels(["0.10","0.25","0.50","0.75","0.90"])
    plt.grid(axis="y")
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylim([0.3, 0.8])
    plt.ylabel(label, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Dual_accuracy_weight_"+m))
    fig.show()

def plot_confusion_timestep():
    model_name = "/Volumes/DropSave/Tese/trainedModels/dual/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_90/testing"
    #model_name = "/Volumes/DropSave/Tese/trainedModels/dual/Concat/fov75_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_90/testing"
    detect_by_class = []

    when = [0, 0, 0, 0, 0, 0, 0]
    count_all = 0
    count_TA_TP = 0

    test_dataset_dir = ["/Volumes/DropSave/Tese/dataset/test_dictionary.json",
                        "/Volumes/DropSave/Tese/dataset/test_dictionary_TA.json"]
    with open(test_dataset_dir[0]) as fp:
        test_dict = json.load(fp)
    with open(test_dataset_dir[1]) as fp:
        test_dict2 = json.load(fp)

    for k in test_dict2.keys():
        if k not in test_dict:
            test_dict[k] = test_dict2[k]
        else:
            for task in test_dict2[k].keys():
                test_dict[k][task] = test_dict2[k][task]

    sequence = []
    for i in range(7):
        sequence.append([0, 0, 0, 0])  # TP, TN, FP, FN

    count_TA = 0
    count_TP = 0
    in_sequence = 0

    for idx_task, task in enumerate(config.classes):
        # print(task)
        detect_by_class.append([])
        # when = [0, 0, 0, 0, 0, 0, 0]
        # TP, TN, FP, FN = [0, 0, 0, 0]
        total_task = 0

        for test in glob(os.path.join(model_name, task, "*")):
            detect_by_class.append([])
            total_task = 0
            count_TA_TP += 1

            with np.load(test) as f:
                seq = f["seqs"][0]
                detect = f["detects"][0]
                all_detect = f["detects"]
                detect_by_class[idx_task].append(detect)

            f_dir, f_name = os.path.split(test)
            image_name = f_name.split(".")[0]

            if len(test_dict[image_name][task]) != 0:
                bx, by, bw, bh = test_dict[image_name][task]
                count_all += 1
                total_task += 1
                count_TP += 1
            else:
                bx, by, bw, bh = [-1, -1, 0, 0]
                count_TA += 1

            flag = 0
            for idx, fp in enumerate(seq):
                grid_x, grid_y = ind2gridcoord(fp)
                x, y = gridcoord2realcoord(grid_x, grid_y)
                if fixated_object(x, y, bx, by, bw, bh):
                    if not flag:
                        when[idx] += 1
                        in_sequence += 1
                        flag = 1
                    if detect[idx][0] > 0.5:
                        sequence[idx][0] += 1
                        # TP += 1
                    else:
                        sequence[idx][2] += 1
                        # FN += 1
                else:
                    if detect[idx][0] > 0.5:
                        sequence[idx][1] += 1
                        # FP += 1
                    else:
                        sequence[idx][3] += 1
                        # TN += 1

    confusion = [[], []]
    for idx in range(7):
        confusion[0].extend([sequence[idx][0], sequence[idx][1]])
        confusion[1].extend([sequence[idx][2], sequence[idx][3]])

    for t in range(7):
        for i in range(2):
            for j in range(2):
                confusion[i][t * 2 + j] = int(round(100.0 * confusion[i][t * 2 + j] / float(count_TA_TP)))
                print(i, t * 2 + j)

    # im = ax.imshow(confusion)

    ax = []
    '''for j in range(2):
        for i in range(4):
            if j == 1 and i == 3:
                continue
            else:
                ax.append(plt.subplot2grid((2, 4), (j, i)))'''

    fig = plt.figure(figsize=(10, 3), dpi=80)
    for j in range(7):
        ax.append(plt.subplot2grid((1, 7), (0, j)))

    '''for i in range(4):
        for j in range(2):
            idx = j * 4 + i
            if idx >= 7:
                continue
            ax[idx].imshow([[sequence[idx][0], sequence[idx][3]], [sequence[idx][2], sequence[idx][1]]])'''
    for idx in range(7):
        ax[idx].imshow([[sequence[idx][0], sequence[idx][1]], [sequence[idx][2], sequence[idx][3]]], cmap='Blues')
        ax[idx].set_xticks([0, 1])
        ax[idx].set_xticklabels([1, 0], fontsize=12)
        ax[idx].set_yticks([0, 1])
        ax[idx].set_yticklabels([1, 0], fontsize=12)
        ax[idx].set_title("t = " + str(idx), fontsize=12)
        for i in range(2):
            for j in range(2):
                color = "slategrey"
                if confusion[i][2 * idx + j] > 26:
                    color = 'white'
                text = ax[idx].text(j, i, confusion[i][2 * idx + j],
                                    ha="center", va="center", color=color)

    # fig.suptitle("Concatenation Architecture", fontsize=18)
    ax[3].set_xlabel("Actual Values", fontsize=14)
    ax[0].set_ylabel("Predicted Values", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(save_dir, "dual_fixFirst_concat_v2"))
    plt.show()

    print(count_TA)
    print(count_TP)
    print(in_sequence)

def get_TFP_Panop():
    _, human_all, _ = getWhenFixatedTrainingDatset()

    human = np.mean(human_all, axis=0)
    human_err = np.std(human_all, axis=0)

    panop_path = "/Volumes/DropSave/Tese/trainedModels/hardPanop_sig2_r1accum_normal_rnny_onehot_label_redDepth_1_act_sigm/testing"
    vgg_path = "/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing"

    print("Executing panop")
    panop_all = getCumulTFPTestModel(panop_path)
    panop = np.mean(panop_all, axis=0)
    panop_err = np.std(panop_all, axis=0)
    panop_err = []
    for t in range(7):
        panop_err.append(std(panop_all[:, t]))
    panop_err = np.array(panop_err)

    print("Executing vgg")
    vgg_all = getCumulTFPTestModel(vgg_path)
    vgg = np.mean(vgg_all, axis=0)
    vgg_err = []
    for t in range(7):
        vgg_err.append(std(vgg_all[:, t]))
    vgg_err = np.array(vgg_err)

    print("Executing rand")
    rand_path = "/Volumes/DropSave/Tese/trainedModels/random/testing"
    rand_all = getCumulTFPTestModel(rand_path)
    rand = np.mean(rand_all, axis=0)
    rand_err = np.std(rand_all, axis=0)
    rand_err = []
    for t in range(7):
        rand_err.append(std(rand_all[:, t]))
    rand_err = np.array(rand_err)

    x = range(7)

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.errorbar(x, human, yerr=human_err, label="Human", ls="--")
    plt.errorbar(x, vgg, yerr=vgg_err, label="ConvLSTM - High-level Features", )
    plt.errorbar(x, panop, yerr=panop_err, label="ConvLSTM - Panoptic Features")
    plt.errorbar(x, rand, yerr=rand_err, label="Random scanpath")
    plt.xlim([0, 6])
    plt.ylim([0, 1])
    plt.xlabel("Time step")
    plt.ylabel("Target fixation cumulative probability")
    box = ax.get_position()
    '''ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])'''

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", ncol=2, mode="expand")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Panop_TFP"))
    plt.show()

    test_data = json.load(open("/Volumes/DropSave/Tese/dataset/test_dictionary.json"))

    ratios = []
    for task_path in glob(os.path.join(panop_path, "*")):
        task_ratios = []
        for exp in glob(os.path.join(task_path, "*")):
            task = exp.split("/")[-2]
            img = exp.split("/")[-1].split(".")[0]
            bbox = test_data[img][task]
            with np.load(exp) as data:
                seq = data["seqs"][0]

            task_ratios.append(scanpath_ratio_func(seq, bbox))
        ratios.append(np.mean(task_ratios))


    print("Ratios", np.mean(ratios))
    #human = getWhenFixatedTrainingDatset()
    human_mean = human.mean(axis=0)
    aux = np.abs(human_mean - panop)
    print("Mismatch", np.sum(aux))
    #plot_bars_distance_length()



    return np.array(ratios)

def get_TFP_dual():
    concat_path = "/Volumes/DropSave/Tese/trainedModels/dual/Concat/fov75_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_90/testing"
    fix_first_path = "/Volumes/DropSave/Tese/trainedModels/dual/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_50/testing"
    _, human_all, _ = getWhenFixatedTrainingDatset()

    human = np.mean(human_all, axis=0)
    human_err = np.std(human_all, axis=0)

    panop_path = "/Volumes/DropSave/Tese/trainedModels/hardPanop_sig2_r1accum_normal_rnny_onehot_label_redDepth_1_act_sigm/testing"
    vgg_path = "/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing"
    beatriz_path = "/Users/beatrizpaula/Downloads/fov100_batch32_onehot_rnny_onehot_label/testing"

    print("Executing panop")
    panop_all = getCumulTFPTestModel(panop_path)
    panop = np.mean(panop_all, axis=0)
    panop_err = np.std(panop_all, axis=0)
    panop_err = []
    for t in range(7):
        panop_err.append(std(panop_all[:, t]))
    panop_err = np.array(panop_err)

    print("Executing vgg")
    vgg_all = getCumulTFPTestModel(vgg_path)
    vgg = np.mean(vgg_all, axis=0)
    vgg_err = []
    for t in range(7):
        vgg_err.append(std(vgg_all[:, t]))
    vgg_err = np.array(vgg_err)
    beatriz_all = getCumulTFPTestModel(beatriz_path)
    beatriz = np.mean(beatriz_all, axis=0)
    beatriz_err = []
    for t in range(7):
        beatriz_err.append(std(beatriz_all[:, t]))
    beatriz_err = np.array(beatriz_err)

    print("Executing concat")
    concat_all = getCumulTFPTestModel(concat_path, dual=1)
    concat = np.mean(concat_all, axis=0)
    concat_err = []
    for t in range(7):
        concat_err.append(std(concat_all[:, t]))
    concat_err = np.array(concat_err)

    print("Executing fix")
    fix_first_all = getCumulTFPTestModel(fix_first_path, dual=1)
    fix_first = np.mean(fix_first_all, axis=0)
    fix_first_err = []
    for t in range(7):
        fix_first_err.append(std(fix_first_all[:, t]))
    fix_first_err = np.array(fix_first_err)


    rand_path = "/Volumes/DropSave/Tese/trainedModels/random/testing"
    rand_all = getCumulTFPTestModel(rand_path)
    rand = np.mean(rand_all, axis=0)
    rand_err = np.std(rand_all, axis=0)
    rand_err = []
    for t in range(7):
        rand_err.append(std(rand_all[:, t]))
    rand_err = np.array(rand_err)

    x = range(7)

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.errorbar(x, human, yerr=human_err, label="Human", ls="--")
    plt.errorbar(x, vgg, yerr=vgg_err, label="High-level Features - Gaussian GT")
    plt.errorbar(x, beatriz, yerr=beatriz_err, label="High-level Features - One-hot GT")
    plt.errorbar(x, panop, yerr=panop_err, label="Panoptic Features")
    plt.errorbar(x, rand, yerr=rand_err, label="Random scanpath")
    plt.errorbar(x, fix_first, yerr=panop_err, label="Dual - Architecture A")
    plt.errorbar(x, concat, yerr=rand_err, label="Dual - Architecture C")
    plt.xlim([0, 6])
    plt.ylim([0, 1])
    plt.xlabel("Time step")
    plt.ylabel("Target fixation cumulative probability")
    box = ax.get_position()
    '''ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])'''

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", ncol=2, mode="expand")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ALL_TFP"))
    plt.show()

    '''# dictionary with object bbox info
    test_dataset_dir = ["/Volumes/DropSave/Tese/dataset/test_dictionary.json",
                        "/Volumes/DropSave/Tese/dataset/test_dictionary_TA.json"]
    with open(test_dataset_dir[0]) as fp:
        test_dict = json.load(fp)
    with open(test_dataset_dir[1]) as fp:
        test_dict2 = json.load(fp)

    for k in test_dict2.keys():
        if k not in test_dict:
            test_dict[k] = test_dict2[k]
        else:
            for task in test_dict2[k].keys():
                test_dict[k][task] = test_dict2[k][task]

    print("Ratios")
    ratios = []
    human_mean = human.mean(axis=0)
    for idx_path, path in enumerate([fix_first_path, concat_path]):
        for task_path in glob(os.path.join(path, "*")):
            task_ratios = []
            for exp in glob(os.path.join(task_path, "*")):
                task = exp.split("/")[-2]
                img = exp.split("/")[-1].split(".")[0]
                if len(test_dict[img][task]) != 0:
                    bbox = test_dict[img][task]
                else:
                    continue
                with np.load(exp) as data:
                    seq = data["seqs"][0]
                task_ratios.append(scanpath_ratio_func(seq, bbox))
            ratios.append(np.mean(task_ratios))

        print(idx_path, "Ratios", np.mean(ratios))

    # human = getWhenFixatedTrainingDatset()
    for idx, data in enumerate([fix_first, concat]):
        aux = np.abs(human_mean - data)
        print(idx, "Mismatch", np.sum(aux))
        print(idx, "Area", get_AUC(data))

    #plot_bars_distance_length()

    return np.array(ratios)'''

if __name__ == "__main__":
    #ratios = get_TFP_dual()

    file = "/Users/beatrizpaula/Desktop/Tese/Dual_results_discrete_twoconvs.txt"

    data = []
    metrics = {"fix_acc": [], "det_acc": [], "w_acc": []}
    keys = []

    path = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"

    with open(path) as fp:
        data = json.load(fp)

    imgs = {}
    bboxs = []

    mult = dict()
    for c in config.classes:
        mult[c] = dict()
    for img in data.keys():

        if len(data[img].keys()) > 1:
            for task1 in data[img].keys():
                for task2 in data[img].keys():
                    if task1 == task2: continue
                    if task2 in mult[task1].keys():
                        mult[task1][task2] += 1
                    else:
                        mult[task1][task2] = 1
                    imgs[img+".npz"] = {task1: data[img][task1], task2: data[img][task2]}


            '''if "tv" in data[img].keys() and "chair" in data[img].keys():
                imgs.append(img+".npz")
                bbox = {"tv": data[img]["tv"], "chair": data[img]["chair"]}
                bboxs.append(bbox)'''

    model = "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label/testing"
    for img_idx, img in enumerate(imgs.keys()):
        print(img)
        for task in imgs[img].keys():
            detect = []
            data_path = os.path.join(model, task, img)
            with np.load(data_path, allow_pickle=True) as fp:
                seq = fp["seqs"][0]
            bbox = imgs[img][task]
            fps = bbox2inds(bbox)
            for t in seq:
                detect.append(int(t in fps))
            print(task, detect)
    '''
   # Boxplots dual
    with open(file, encoding="utf-8") as fobj:
        for idx, line in enumerate(fobj):
            if idx == 0:
                continue
            row = line.split(", ")
            data.append(row)
            metrics["fix_acc"].append(float(row[-3]))
            metrics["det_acc"].append(float(row[-2]))
            metrics["w_acc"].append(float(row[-1]))
            var = []
            var.extend(row[:3])
            keys.append(var)

    plot_dual_boxplots(metrics, keys)'''
    #get_TFP_dual()
    #plot_bars_distance_length()




    '''#Confusion Dual detection
    plot_confusion_timestep()'''

    '''#Compare configs Dual
    data = []
    metrics = {"fix_acc": [], "det_acc": [], "w_acc": []}
    keys = []

    file = "/Users/beatrizpaula/Desktop/Tese/Dual_results.txt"
    with open(file, encoding="utf-8") as fobj:
        for idx, line in enumerate(fobj):
            if idx == 0:
                continue
            row = line.split(", ")
            data.append(row)
            metrics["w_acc"].append(float(row[-1]))
            metrics["det_acc"].append(float(row[-2]))
            metrics["fix_acc"].append(float(row[-3]))

            var = []
            var.extend(row[:3])
            keys.append(var)
    plot_dual_boxplots(metrics, keys)'''

    '''#Compare category accuracies
    vgg_path = "/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing"
    vgg_cumul = getCumulTFPTestModel(vgg_path)
    vgg_cumul = vgg_cumul[:,-1]

    panop_path = "/Volumes/DropSave/Tese/trainedModels/hardPanop_sig2_r1accum_normal_rnny_onehot_label_redDepth_1_act_sigm/testing"
    panop_cumul = getCumulTFPTestModel(panop_path)
    panop_cumul = panop_cumul[:,-1]

    print(panop_cumul)
    barWidth = 0.4
    br1 = np.arange(len(vgg_cumul))
    br2 = [x + barWidth for x in br1]


    ax, fig = plt.subplots()
    plt.bar(br1, vgg_cumul, barWidth)
    plt.bar(br2, panop_cumul, barWidth)

    plt.xticks(range(len(vgg_cumul)), config.classes, fontsize=12, rotation=90)
    plt.legend(['VGG', 'Panop'])
    plt.xlabel("Classes", fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=14)
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots()

    vgg_cumul = [0.6895, 0.6650, 0.6601, 0.6520, 0.6520, 0.6324, 0.6127, 0.6046, 0.5850, 0.5752, 0.5605, 0.5490]
    panop_cumul = [0.629084967, 0.635620915, 0.62254902, 0.68627451, 0.678104575, 0.673202614, 0.614379085, 0.60620915, 0.583333333, 0.607843137, 0.593137255, 0.620915033]

    ax.boxplot([vgg_cumul, panop_cumul], showmeans=True, showfliers=True)
    ax.set_xticklabels(["VGG", "Panop"])
    plt.grid(axis="y")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Search Accuracy", fontsize=16)
    plt.tight_layout()
    plt.show()'''






    '''# Plot panop boxplots
    with open("/Users/beatrizpaula/Desktop/FP2_all_metrics.json") as fp:
        data = json.load(fp)
    plot_Panop_boxplots(data)
    # Get VGG Table data
    

    beatriz_path = "/Users/beatrizpaula/Downloads/fov100_batch32_onehot_rnny_onehot_label/testing"
    meu_path = "/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing"
    rand_path = "/Volumes/DropSave/Tese/trainedModels/random/testing"

    test_data = json.load(open("/Volumes/DropSave/Tese/dataset/test_dictionary.json"))

    ratios = []
    for task_path in glob(os.path.join(beatriz_path, "*")):
        task_ratios=[]
        for exp in glob(os.path.join(task_path, "*")):
            task = exp.split("/")[-2]
            img = exp.split("/")[-1].split(".")[0]
            bbox = test_data[img][task]
            with np.load(exp) as data:
                seq = data["seqs"][0]

            task_ratios.append(scanpath_ratio_func(seq, bbox))
        ratios.append(np.mean(task_ratios))

    print(np.mean(ratios))

    plot_bars_distance_length()'''






    '''path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"

    with open(path) as fp:
        training_data = json.load(fp)

    ratios = []

    for obs in training_data:
        flag = 0
        for i_fixated in range(obs["length"]):
            if fixated_object(obs["X"][i_fixated], obs["Y"][i_fixated], obs["bbox"][0], obs["bbox"][1], obs["bbox"][2],
                              obs["bbox"][3]):
                flag = 1
                break

        if not flag: continue

        bbox = obs["bbox"]
        seq = []
        for i in range(min(7, obs["length"])):
            x, y = realcoord2gridcoord(obs['X'][i], obs['Y'][i])
            seq.append(gridcoord2ind(x, y))
        ratios.append(scanpath_ratio_func(seq, bbox))

    final_ratio = np.mean(ratios)

    human = getWhenFixatedTrainingDatset()
    human_mean = human[1].mean(axis=0)

    beatriz_path = "/Users/beatrizpaula/Downloads/fov100_batch32_onehot_rnny_onehot_label/testing"
    meu_path = "/Users/beatrizpaula/Downloads/fov100_batch256_normal_rnny_onehot_label/testing"

    beatriz_all = getCumulTFPTestModel(beatriz_path)
    beatriz = np.mean(beatriz_all, axis=0)
    beatriz_err = np.std(beatriz_all, axis=0)
    beatriz_err = []
    for t in range(7):
        beatriz_err.append(std(beatriz_all[:, t]))
    beatriz_err = np.array(beatriz_err)

    meu_all = getCumulTFPTestModel(meu_path)
    meu = np.mean(meu_all, axis=0)
    meu_err = []
    for t in range(7):
        meu_err.append(std(meu_all[:, t]))
    meu_err = np.array(meu_err)

    rand_path = "/Volumes/DropSave/Tese/trainedModels/random/testing"
    rand_all = getCumulTFPTestModel(rand_path)
    rand = np.mean(rand_all, axis=0)

    meu_area = get_AUC(meu)
    beatriz_area = get_AUC(beatriz)
    random_area = get_AUC(rand)

    aux = np.abs(human_mean-meu)
    print(np.sum(aux))

    aux = np.abs(human_mean - beatriz)
    print(np.sum(aux))

    aux = np.abs(human_mean - rand)
    print(np.sum(aux))'''






    '''p = "/Users/beatrizpaula/Desktop/aux2.txt"

    foveas = [100, 75, 50]
    tasks = ["One-hot", "Heatmap-1d"]
    gts = ["One-hot", "Gaussian"]

    d = dict()

    fp = open(p)

    for fovea in foveas:
        for task in tasks:
            for gt in gts:
                k = (fovea, task, gt)
                comul = fp.readline().split()
                d[k] = [eval(i) for i in comul]

    for enc in d.keys():
        plt.plot(range(7), d[enc], label=str(enc))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    plt.show()'''







    '''# Generate Images per class chart
    values = getImagesPerClass()

    labels = config.classes.copy()
    labels.append("category mean")
    plotBarChart(labels, values)

    # Generate Comul Prob chart
    when, comul, comul2 = getWhenFixatedTrainingDatset()
    x_axis = []
    color = []
    aux = plt.get_cmap('tab20').colors
    for i in range(19):
        x_axis.append(list(range(7)))
        color.append(aux[i])

    color[-1] = 'black'
    xlabel = "Number of fixations made"
    ylabel = "Cumulative probability of target fixation"
    plotLineChart(x_axis, comul, labels, color, xlabel, ylabel)'''
    '''# Excel table
    file = "/Users/beatrizpaula/Desktop/pandasExcel.txt"

    data = []
    metrics = {"top-1": [], "top-3": [],"top-5": [], "area": []}
    keys = []

    with open(file, encoding="utf-16") as fobj:
        for idx, line in enumerate(fobj):
            if idx == 0:
                continue
            row = line.split()
            data.append(row)
            metrics["area"].append(float(row[-1]))
            metrics["top-5"].append(float(row[-2]))
            metrics["top-3"].append(float(row[-3]))
            metrics["top-1"].append(float(row[-4]))
            var = []
            var.extend(row[:4])
            keys.append(var)

    plot_VGG_boxplots(metrics, keys)
    get_TFP_VGG()'''

























