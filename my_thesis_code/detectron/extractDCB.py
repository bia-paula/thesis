import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageFilter, ImageDraw
import detectron2
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import time

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import smooth_foveate

#device = torch.device("cuda")


def pred2feat(seg, info):
    seg = seg.cpu()
    feat = torch.zeros([80 + 54, 320, 512])
    for pred in info:
        mask = (seg == pred['id']).float()
        if pred['isthing']:
            feat[pred['category_id'], :, :] = mask * pred['score']
        else:
            feat[pred['category_id'] + 80, :, :] = mask
    return F.interpolate(feat.unsqueeze(0), size=[20, 32]).squeeze(0)


def get_DCBs(img_path, predictor, radius=1):
    high = Image.open(img_path).convert('RGB').resize((512, 320))
    low = high.filter(ImageFilter.GaussianBlur(radius=radius))
    high_panoptic_seg, high_segments_info = predictor(
        np.array(high))["panoptic_seg"]
    low_panoptic_seg, low_segments_info = predictor(
        np.array(low))["panoptic_seg"]
    high_feat = pred2feat(high_panoptic_seg, high_segments_info)
    low_feat = pred2feat(low_panoptic_seg, low_segments_info)
    return high_feat, low_feat


if __name__ == '__main__':

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

    img_path = "/Volumes/DropSave/Tese/dataset/resized_images/chair/000000054598.jpg"

    fovea_size = 50

    high_feat, low_feat = get_DCBs(img_path, predictor)

    print(high_feat.shape, low_feat.shape)

    high = Image.open(img_path).convert('RGB').resize((512, 320))
    low = high.filter(ImageFilter.GaussianBlur(radius=2))
    low2 = high.filter(ImageFilter.GaussianBlur(radius=7))
    #low.show()
    #high.show()

    img_array = img_to_array(high)

    smooth = smooth_foveate(img_array, x=256, y=160, fovea_size=fovea_size)
    array_to_img(smooth).show()

    time.sleep(2)


    x = 256
    y = 160

    w = 512
    h = 320

    r = fovea_size

    Y, X = np.ogrid[:h, :w]

    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    mask = dist <= r
    ver_channel = mask.astype(np.float32)
    ver = np.atleast_3d(ver_channel)



    #mask = torch.from_numpy(mask)
    #mask = mask.unsqueeze(0).repeat(320, 512, 3)
    low = (1 - ver) * low + ver * high
    low2 = (1 - ver) * low2 + ver * high

    img_hard_bf = array_to_img(low)
    img_hard_after = ImageDraw.Draw(img_hard_bf)
    center = (256, 160)
    r = 50
    shape = [center[0] - r, center[1] - r, center[0] + r, center[1] + r]
    #img_hard_after.ellipse(shape, outline="red")
    img_hard_bf.show()
    time.sleep(2)

    image_hard2 = array_to_img(low2)
    image_hard2.show()
    time.sleep(2)



    seg, info = predictor(smooth)["panoptic_seg"]

    v = Visualizer(smooth, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    draw_fov = v.draw_panoptic_seg(seg, info)
    array_to_img(draw_fov.get_image()[:, :, ::-1]).show(title="high")

    seg, info = predictor(img_to_array(img_hard_bf))["panoptic_seg"]

    v = Visualizer(img_to_array(img_hard_bf), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    draw_fov = v.draw_panoptic_seg(seg, info)
    array_to_img(draw_fov.get_image()[:, :, ::-1]).show(title="img_hard_bf")


    seg, info = predictor(img_to_array(image_hard2))["panoptic_seg"]

    v = Visualizer(img_to_array(image_hard2), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    draw_fov = v.draw_panoptic_seg(seg, info)
    array_to_img(draw_fov.get_image()[:, :, ::-1]).show(title="image_hard2")
    

    # Compute DCB
    #img_path = '/home/zhibyang/projects/datasets/coco_search/images/320x512/TP/bottle/000000573206.jpg'
    #high_feat, low_feat = get_DCBs(img_path, predictor)
    #print(high_feat.shape, low_feat.shape)

