# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import torch
import math
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

# Reading an image in default mode
im = cv2.imread("000000578751.jpg")
# Window name in which image is displayed
window_name = 'image'

#cv2.imshow(window_name, im)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
#cv2.waitKey()

# closing all open windows
#cv2.destroyAllWindows()

cfg = get_cfg()
print(torch.backends.mps.is_available())
cfg.MODEL.DEVICE = 'cpu'
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")

predictor = DefaultPredictor(cfg)

predictions = predictor(im)

print(predictions["sem_seg"])

'''
# Build the model from config
model = build_model(cfg)
model.eval()

# Loading an image just for testing
im = cv2.imread("000000578751.jpg")
im_shape = [im.shape[:2]]
print(im.shape)
im_tensor = torch.from_numpy(im).permute(2, 0, 1).float()
print(torch.max(im_tensor))
tensor_list = [im_tensor]

images = ImageList.from_tensors(tensor_list)

out1 = model.backbone(images.tensor)
features = model.backbone(images.tensor)
out2 = model.proposal_generator(images, features)
proposals, _ = model.proposal_generator(images, features)
out3 = model.roi_heads(images, features, proposals)
instances, _ = model.roi_heads(images, features, proposals)
mask_features = [features[f] for f in model.roi_heads.in_features]
mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])


#v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#cv2.imshow(window_name, out.get_image()[:, :, ::-1])
#cv2.waitKey()
#cv2.destroyAllWindows()

#aqui = model_zoo.get_config("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")'''

