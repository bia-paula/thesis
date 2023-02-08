import os.path

import numpy as np
import math
import torch
import torch.nn.functional as F

from time import time
import pickle

from glob import glob
import os

import sys

sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config


def get_features(panop_out):
    seg, info = panop_out
    feat = torch.zeros([80 + 54, 320, 512])
    for pred in info:
        mask = (seg == pred['id']).float()
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat[pred['category_id'] + 80, :, :] = mask
    return torch.permute(F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0), (1, 2, 0))


if __name__ == "__main__":

    save_dir = "/Users/beatrizpaula/Desktop/images_HL_DCBs/DCBs"
    load_dir = "/Users/beatrizpaula/Desktop/images_HL_DCBs/high_low"

    for category_path in glob(os.path.join(load_dir, "*")):
        category = category_path.split("/")[-1]
        print("Starting", category)

        try:
            os.mkdir(os.path.join(save_dir, category))
        except FileExistsError:
            print("Saving Category Directory already exists...")

        for load_path in glob(os.path.join(category_path, "*")):
            file_name = load_path.split("/")[-1]

            with open(load_path, "rb") as fp:
                panop_out = pickle.load(fp)

            feat = np.array(get_features(panop_out))

            np.savez(os.path.join(save_dir, category, file_name), dcb=feat)





