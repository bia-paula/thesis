from glob import glob
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import config
from vgg16_predict import classes_dict


# Function to calculate True Positive Rate and False Positive Rate

def calc_TP_FP_rate(y_true, y_pred):

    # Instantiate counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    # Calculate true positive rate and false positive rate
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)

    return tpr, fpr

def calc_P_R_rate(y_true, y_pred):

    # Instantiate counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1

    # Calculate true positive rate and false positive rate
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    #print(TP, FP, TN, FN)

    if (TP + FP) == 0:
        p = 1
    else:
        p = TP / (TP + FP)
    r = TP / (TP + FN)

    return p, r

if __name__ == '__main__':

    fovea = 50

    model_name = "fov100_batch256_normal_rnny_onehot_label"
    save_dir = "/Users/beatrizpaula/Desktop/Tese/images/baselineVGG"
    paths = glob(os.path.join("/Volumes/DropSave/Tese/trainedModels", model_name, "detection_preTrained", "*fov"+str(fovea)+"*.json"))

    TrueAndFalses = [0, 0, 0, 0]
    precision = []
    recall = []
    accuracy = []

    positives = []
    negatives = []

    check = np.zeros((18, 2))

    confusion = dict()
    metrics = {"accuracy": [], "precision": [], "recall": []}
    results = dict()
    for task in config.classes:
        confusion[task] = {"TP": 0, "FP": 0, "TN": 0, "FN":0}
        results[task] = {"true":[], "pred":[]}

    classes_in = []
    for task_idx, task in enumerate(paths):
        help = 0
        dir, file_name = os.path.split(task)
        if "detection_probabilities_fov" in file_name:
            cat = "ALL"
        else:
            #cat = file_name.split(".")[0].split("_")[-1]
            cat = file_name.split(".")[0].split("_")[-2]
            task_idx = config.classes.index(cat)

        with open(task) as fp:
            exps = json.load(fp)

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        if cat != "ALL":
            pred = exps["prediction"]
            gt = exps["gt"]
            if len(pred) == 0:
                continue
            for idx in range(len(pred)):
                results[cat]["true"].extend(gt[idx])
                results[cat]["pred"].extend(pred[idx])
                for fp in range(len(pred[idx])):
                    help = 1
                    # Check actual OBJECT DETECTED
                    if gt[idx][fp]:
                        if pred[idx][fp] > 0.5:
                            TP += 1
                            confusion[cat]["TP"] += 1
                            check[task_idx, 1] += 1
                        else:
                            FN += 1
                            confusion[cat]["FN"] += 1
                            check[task_idx, 0] += 1
                    # Check actual OBJECT NOT DETECTED
                    else:
                        if pred[idx][fp] <= 0.5:
                            TN += 1
                            confusion[cat]["TN"] += 1
                            check[task_idx, 0] += 1
                        else:
                            FP += 1
                            confusion[cat]["FP"] += 1
                            check[task_idx, 1] += 1

            if help: classes_in.append(cat)
            TrueAndFalses[0] += TP
            TrueAndFalses[1] += FP
            TrueAndFalses[2] += TN
            TrueAndFalses[3] += FN

            print("***", cat, "***")
            print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
            if TP + TN + FP + FN == 0:
                print("NOT PRESENT")
                continue
            try:
                prec = TP / (TP + FP)
            except ZeroDivisionError:
                prec = np.NAN
            precision.append(prec)
            print("Precision:", prec)
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                rec = np.NAN
            recall.append(rec)
            print("Recall:", rec)
            accuracy.append((TP + TN) / (TP + TN + FP + FN))
            print("Accuracy:", (TP + TN) / (TP + TN + FP + FN), "\n")


            positives.append(TP+FN)
            negatives.append(TN+FP)


        '''else:
            for t in exps.keys():
                pred = exps[t]["prediction"]
                gt = exps[t]["gt"]

                for idx in range(len(t)):
                    for fp in range(7):
                        # Check actual OBJECT DETECTED
                        if gt[idx][fp]:
                            if pred[idx][fp] > 0.5:
                                TP += 1
                            else:
                                FN += 1
                        # Check actual OBJECT NOT DETECTED
                        else:
                            if pred[idx][fp] <= 0.5:
                                TN += 1
                            else:
                                FP += 1'''

    print("***", "ALL", "***")
    TP, FP, TN, FN = TrueAndFalses
    print("TP:", TP, "FP:", FP, "TN:", TN, "FN:", FN)
    print("Precision:", np.nanmean(precision), "+-", np.std(precision))
    print("Recall:", np.nanmean(recall), "+-", np.std(recall))
    print("Accuracy:", np.nanmean(accuracy), "+-", np.std(accuracy), "\n")
    precision.append(np.nanmean(precision))
    recall.append(np.nanmean(recall))
    accuracy.append(np.nanmean(accuracy))
    classes_in.append("mean")


    '''for c in config.classes:
        TP, TN, FP, FN = confusion[c]["TP"], confusion[c]["TN"], confusion[c]["FP"], confusion[c]["FN"]
        if TP + TN + FP + FN == 0:
            continue
        classe_present.append(c)
        metrics["accuracy"].append((TP + TN)/(TP + TN + FP + FN))
        if TP + FP != 0:
            prec = TP / (TP + FP)
        else:
            prec = 1
        metrics["precision"].append(prec)
        if TP + FN != 0:
            rec = TP / (TP + FN)
        else:
            rec = 1
        metrics["recall"].append(rec)

    #ADD MEAN OVER CLASSES
    metrics["accuracy"].append(np.mean(metrics["accuracy"]))
    metrics["precision"].append(np.mean(metrics["precision"]))
    metrics["recall"].append(np.mean(metrics["recall"]))
    positives.append(np.mean(positives))
    negatives.append(np.mean(negatives))'''

    markers = ["o", "s", "^"]
    x = classes_in.copy()
    #x = classe_present.copy()
    #x.append("mean")
    y = [metrics["accuracy"], metrics["precision"],  metrics["recall"]]
    y = [accuracy, precision, recall]
    c = ["dodgerblue", "darkorange", "yellowgreen"]

    fig, ax = plt.subplots()
    ms = []
    for yp, m, cp in zip(y, markers, c):
        new = ax.scatter(x, yp, marker=m, c=cp)
        ms.append(new)

    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="both")
    plt.ylim([0,1])
    plt.tight_layout()
    ax.legend(handles=ms, labels=["Accuracy", "Precision", "Recall"], fontsize=14)
    plt.savefig(os.path.join(save_dir, "detection_metrics_fov"+str(fovea)))
    plt.show()
    print("FOVEA", fovea)
    '''
    fig, ax = plt.subplots()

    for i in range(len(positives)):
        t = positives[i] + negatives[i]
        positives[i] = positives[i]/t
        negatives[i] = negatives[i] / t

    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)

    print("FOVEA", fovea)

    #IMAGES RATIO
    width = 1
    b1 = ax.bar(x, positives, width, label='Positive', color=["#92D050"], alpha=0.8)
    b2 = ax.bar(x, negatives, width,  bottom=positives,
           label='Negative', color=["#FF2600"], alpha=0.8)
    plt.legend([b1, b2], ["Present", "Not Present"], fontsize=14)
    plt.xlabel("Classes", fontsize=16)
    plt.ylabel("Ratio of Images", fontsize=16)
    plt.tight_layout()
    plt.grid(axis="y")
    plt.show()

    print(np.argmin(positives))
    print(max(positives))
    print(positives[-1])'''

    
    '''#ROC CURVES
    probability_thresholds = np.linspace(0, 1, num=1000)
    classes_tp = []
    classes_fp = []
    classes_p = []
    classes_r = []
    for i in probability_thresholds:
        classes_tp.append([])
        classes_fp.append([])
        classes_p.append([])
        classes_r.append([])



    aucs = []
    # Find true positive / false positive rate for each threshold
    for target in config.classes:
        fig, ax = plt.subplots()
        # Containers for true positive / false positive rates
        lr_tp_rates = []
        lr_fp_rates = []
        p_rates = []
        r_rates = []
        for p_idx, p in enumerate(probability_thresholds):

            y_test_preds = []

            for prob in results[target]["pred"]:
                if prob < 0.002:#0.011:
                    prob = 0.002#0.011

                if prob > 0.998:#0.989:
                    prob = 0.998#0.989

                if prob > p:
                    y_test_preds.append(1)
                else:
                    y_test_preds.append(0)

            tp_rate, fp_rate = calc_TP_FP_rate(results[target]["true"], y_test_preds)
            p_rate, r_rate = calc_P_R_rate(results[target]["true"], y_test_preds)

            lr_tp_rates.append(tp_rate)
            lr_fp_rates.append(fp_rate)
            p_rates.append(p_rate)
            r_rates.append(r_rate)

            classes_tp[p_idx].append(tp_rate)
            classes_fp[p_idx].append(fp_rate)
            classes_p.append(p_rate)
            classes_r.append(r_rate)

        print("Class:", target)
        print("TP_rate:", min(lr_tp_rates), "-", max(lr_tp_rates))
        print("FP_rate:", min(lr_fp_rates), "-", max(lr_fp_rates))

        print("Class:", target)
        print("TP_rate:", min(lr_tp_rates), "-", max(lr_tp_rates))
        print("FP_rate:", min(lr_fp_rates), "-", max(lr_fp_rates))

        auc = 0
        for idx, h_aux in enumerate(probability_thresholds[1:]):
            a = lr_tp_rates[idx-1]
            b = lr_tp_rates[idx]
            h = lr_fp_rates[idx]-lr_fp_rates[-1]
            #a = p_rates[idx - 1]
            #b = p_rates[idx]
            #h = r_rates[idx] - r_rates[idx-1]
            auc += 0.5*(a+b)*h





        curve, = ax.plot(lr_fp_rates, lr_tp_rates)
        xdata = curve.get_xdata()
        xdata.sort()
        ydata = curve.get_ydata()
        ydata.sort()

        auc = 0
        for idx in range(len(xdata)-1):
            a = ydata[idx]
            b = ydata[idx+1]
            h = xdata[idx+1] - xdata[idx]
            auc += 0.5 * (a + b) * h

        aucs.append(auc)

        plt.title(target.capitalize() + " (AUC = {:.3f})".format(auc), fontsize=18)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        #plt.savefig(os.path.join(save_dir, "ROC_"+target))
        plt.show()

    classes_tp = np.array(classes_tp)
    classes_fp = np.array(classes_fp)
    classes_tp = np.mean(classes_tp, axis=1)
    classes_fp = np.mean(classes_fp, axis=1)
    fig, ax = plt.subplots()



    curve, = ax.plot(classes_fp, classes_tp)
    xdata = curve.get_xdata()
    ydata = curve.get_ydata()
    xdata.sort()
    ydata.sort()

    auc = 0
    for idx in range(len(xdata)-1):
        a = ydata[idx]
        b = ydata[idx + 1]
        h = xdata[idx +1] - xdata[idx]
        auc += 0.5 * (a + b) * h




    plt.title("Mean over classes (AUC = {:.3f})".format(auc), fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, "PR_AverageMean"))
    plt.show()

    print(config.classes[np.argmin(aucs)], min(aucs))
    print(config.classes[np.argmax(aucs)], max(aucs))

    order = np.argsort(aucs)
    order_classes = []
    for i in order:
        if i == 18:
            continue
        order_classes.append(config.classes[i])'''













