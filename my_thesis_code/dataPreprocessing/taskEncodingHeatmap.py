import json
# from foveateImages import get_class_id, realcoord2gridcoord, gridcoord2ind
import numpy as np
from datetime import timedelta
import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
import time
from glob import glob


def heat_map_encoder(heatmap_type, one_hot, heatmaps):
    id = np.argmax(one_hot)
    if heatmap_type == 1:
        ret = heatmaps[id]
    elif heatmap_type == 2:
        heatmap_flat = heatmaps[id]
        heatmap_flat_imagesize = np.array(np.split(heatmap_flat, 10))
        broadcasted = np.broadcast_to(heatmap_flat_imagesize,
                                      (config.fmap_size[2], config.fmap_size[0], config.fmap_size[1]))
        ret = np.moveaxis(broadcasted, 0, 2)

    return ret

    #id = np.argmax(one_hot)
    #return heatmaps[id]


'''
def heat_map_encoder(one_hot, fmapSize, heatmaps):
    id = np.argmax(one_hot)
    if fmapSize:
        heatmap_flat = heatmaps[id]
        heatmap_flat_imagesize = np.array(np.split(heatmap_flat, 10))
        broadcasted = np.broadcast_to(heatmap_flat_imagesize,
                                      (config.fmap_size[2], config.fmap_size[0], config.fmap_size[1]))
        ret = np.moveaxis(broadcasted, 0, 2)
    else:
        ret = heatmaps[id]
    return ret'''

if __name__ == "__main__":

    '''train_path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
    with open(train_path) as file:
        data = json.load(file)

    shape = (len(config.classes), config.fmap_size[0] * config.fmap_size[1])
    task_heatmaps = np.zeros(shape)
    for count,obs in enumerate(data):
        task_id = get_class_id(obs["task"])
        for fp in range(obs["length"]):
            x_grid, y_grid = realcoord2gridcoord(obs["X"][fp], obs["Y"][fp])
            idx = gridcoord2ind(x_grid, y_grid)
            task_heatmaps[task_id, idx] += 1
        if count % 100 == 0: print(count,"/",len(data))

    sums0 = np.sum(task_heatmaps, axis=0)
    sums1 = np.sum(task_heatmaps, axis=1)
    test = np.sum(task_heatmaps[0])

    normal_heatmaps = np.zeros(shape)
    for i, sum in enumerate(sums1):
        normal_heatmaps[i, :] = task_heatmaps[i, :] / sum

    print(np.sum(normal_heatmaps[0]))

    save_dir = "/Volumes/DropSave/Tese/dataset/taskencoding_heatmap"
    np.savez_compressed(save_dir, label_heatmap=normal_heatmaps)'''

    heatmap_type = 2

    # Training directories
    for batch_size in [64, 32]:  # <-----------
        print("BATCH SIZE: ", batch_size)
        first_time = time.time()
        local_time = time.ctime(first_time)
        print("Local time:", local_time)
        train_path = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + str(batch_size)
        save_dir = "/Volumes/DropSave/Tese/dataset/smaller_batches_padded_truncated/batchSize" + str(batch_size) + \
                   "/heatmapLEGT/"

        files = glob(train_path + "/*.npz")
        heatmaps_dir = "/Volumes/DropSave/Tese/dataset/taskencoding_heatmap.npz"
        heatmaps = np.load(heatmaps_dir)["label_heatmap"]

        files.sort()

        for count, file in enumerate(files):
            # Load file
            label_enc_onehot = np.load(file)["label_encodings"]

            exps_shape = label_enc_onehot.shape[0]

            # label_enc_heatmap = np.zeros((exps_shape, 160))
            label_enc_heatmap = np.zeros((exps_shape, config.fmap_size[0], config.fmap_size[1], config.fmap_size[2]))

            for obs in range(label_enc_onehot.shape[0]):
                heatmap = heat_map_encoder(heatmap_type, label_enc_onehot[obs], heatmaps)
                #heatmap_flat_imagesize = np.array(np.split(heatmap_flat, 10))
                #broadcasted = np.broadcast_to(heatmap_flat_imagesize,
                #                              (config.fmap_size[2], config.fmap_size[0], config.fmap_size[1]))
                #heatmap_flat_fmapsize = np.moveaxis(broadcasted, 0, 2)

                #label_enc_heatmap[obs] = heatmap_flat_fmapsize
                label_enc_heatmap[obs] = heatmap

            name = file.split("/")[-1] + "_fmapSize"
            save_file = save_dir + name
            np.savez_compressed(save_file, label_encodings_heatmap=label_enc_heatmap)
            if count % 10 == 0: print(count + 1, "/", len(files))

        last_time = time.time()
        print("TOTAL TIME: ")
        print(timedelta(seconds=last_time - first_time))
        local_time = time.ctime(last_time)
