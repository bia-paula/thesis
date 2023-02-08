import matplotlib.pyplot as plt
import seaborn as sns
import json

import numpy as np

from glob import glob
import os
import sys

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
    colors = []
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

    when = np.zeros((19, 7))
    all = []
    comul = np.zeros((19, 7))
    comul2 = np.zeros((7))
    for task in config.classes:
        all.append(0)
        print(task)

    for idx_obs, obs in enumerate(training_data):

        flag = 0
        # Get image
        class_name = obs["task"]
        all[config.classes.index(class_name)] += 1
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
            when[config.classes.index(task), i] += when[config.classes.index(task), i - 1]
        # sum = np.sum(when[config.classes.index(task)])
        print(task)
        print(all[config.classes.index(task)])
        comul[config.classes.index(task)] = when[config.classes.index(task)] / all[config.classes.index(task)]

        when[-1] += (when[config.classes.index(task)] / 18)
        comul[-1] += (comul[config.classes.index(task)] / 18)
    comul[-1] = comul[-1]
    print(comul[-1, -1])

    for i in range(1, 7):
        comul2[i] += comul2[i - 1]
    total = np.sum(all)
    comul2 = comul2 / total

    return when, comul, comul2


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

    # plt.legend()
    plt.xlim(0, 6)
    plt.ylim(0, 1)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "datasetCumulProb"))
    plt.show()


if __name__ == "__main__":

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
    # Excel table
    file = "/Users/beatrizpaula/Desktop/pandasExcel.txt"

    data = []
    metrics = {"top-1": [], "top-3": [], "top-5": [], "area": []}
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

    for k in metrics.keys():
        metrics[k] = np.array(metrics[k])

    fovea_acc = {"50": [], "75": [], "100": []}
    for idx, k in enumerate(keys):
        fovea_acc[k[0]].append(metrics["top-1"][idx])

    fig, ax = plt.subplots()
    ax.boxplot(fovea_acc.values(), showmeans=True, showfliers=True)
    ax.set_xticklabels(fovea_acc.keys())
    plt.grid(axis="y")
    fig.show()

















