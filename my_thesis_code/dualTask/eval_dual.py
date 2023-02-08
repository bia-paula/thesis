from glob import glob
import os
import json

import numpy as np
import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from dataPreprocessing.foveateImages import fixated_object, ind2gridcoord, gridcoord2realcoord
from evaluation.accuracyCOCO import bbox2inds

if __name__ == '__main__':

    model_name = "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_detectFirst/testing"
    names = ["/Volumes/DropSave/Tese/trainedModels/dual/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_detectFirst_binary_dense64_dense32_wfix_10/testing",
             "/Volumes/DropSave/Tese/trainedModels/dual/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_detectFirst_binary_dense64_dense32_wfix_25/testing",
             "/Volumes/DropSave/Tese/trainedModels/dual/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_detectFirst_binary_dense64_dense32_wfix_50/testing"]
    names = glob("/Volumes/DropSave/Tese/trainedModels/dual/Concat/fov*/testing")
    save_dir = "/Users/beatrizpaula/Desktop/Tese/Dual_results_discrete.txt"

    header = ["arch", "fovea", "weight_fix", "fix_acc", "det_acc", "w_acc"]

    lines = []

    names.sort()

    #names = ["/Volumes/DropSave/Tese/trainedModels/dual/Concat/fov100_batch256_normal_rnny_onehot_label_dualLSTM_multiDetect_fixFirst_Concat_binary_dense64_dense32_wfix_75/testing"]
    for model_name in names:
        line = []
        '''if "fixFirst" in model_name:
            line.append("fixFirst")
        else:
            line.append("detectFirst")'''
        line.append("concat")

        type_aux = model_name.split("/")[-2].split("_")
        fov = int(type_aux[0][3:])
        w_fix = int(type_aux[-1])
        line.extend([fov, w_fix])

        print(model_name.split("/")[-2])
        TP, TN, FP, FN = [0, 0, 0, 0]
        precision = []
        recall = []
        accuracy = []

        detect_path = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize256/detect/train_detected_batch256.0.0.npz"
        detect_gt = np.load(detect_path)["rnn_y_det_multi"]
        detect_by_class = []


        when = [0, 0, 0, 0, 0, 0, 0]
        count_all = 0


        # dictionary with object bbox info
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

        fix_acc = []
        det_acc = []
        det_prec = []
        det_rec = []
        for idx_task, task in enumerate(config.classes):
            #print(task)
            detect_by_class.append([])
            #when = [0, 0, 0, 0, 0, 0, 0]
            #TP, TN, FP, FN = [0, 0, 0, 0]
            total_task = 0

            for test in glob(os.path.join(model_name, task, "*")):

                with np.load(test) as f:
                    seq = f["seqs"][0]
                    detect = f["detects"][0]
                    all_detect = f["detects"]
                    detect_by_class[idx_task].append(detect)


                f_dir, f_name = os.path.split(test)
                image_name = f_name.split(".")[0]

                if len(test_dict[image_name][task]) != 0:
                    bbox = test_dict[image_name][task]
                    count_all += 1
                    total_task += 1
                else:
                    bbox = [-1, -1, 0, 0]

                flag = 0
                inds = bbox2inds(bbox)
                for idx, fp in enumerate(seq):
                    #grid_x, grid_y = ind2gridcoord(fp)
                    #x, y = gridcoord2realcoord(grid_x, grid_y)
                    if fp in inds:
                    #if fixated_object(x, y, bx, by, bw, bh):
                        if not flag:
                            when[idx] += 1
                            flag = 1
                        if detect[idx][0] > 0.5:
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if detect[idx][0] > 0.5:
                            FP += 1
                        else:
                            TN += 1

            fix_acc.append(when[6]/total_task)
            det_acc.append((TP + TN) / (TP + TN + FP + FN))
            det_prec.append(TP / (TP + FP))
            det_rec.append(TP / (TP + FN))


        #print(when)
        #print(fix_acc)
        worst = np.argmin(fix_acc)
        #print("Worst:", config.classes[worst], fix_acc[worst])
        best = np.argmax(fix_acc)
        #print("Best:", config.classes[best], fix_acc[best])

        for i in range(1, 7):
            when[i] += when[i - 1]



        final_fix_acc = when[6]/count_all
        final_det_acc = (TP + TN) / (TP + TN + FP + FN)

        print(when)
        print("Fixated Top 1 Accuracy:", final_fix_acc)
        print("Detection:")
        print("Accuracy:", final_det_acc)
        print("Precision:", TP / (TP + FP))
        print("Recall:", TP / (TP + FN))

        print()

        line.extend([final_fix_acc, final_det_acc, 0.5*(final_fix_acc + final_det_acc)])
        lines.append(line)

    with open(save_dir, "w") as fp_results:
        fp_results.write(str(header))
        for line in lines:
            fp_results.write(str(line)+"\n")








