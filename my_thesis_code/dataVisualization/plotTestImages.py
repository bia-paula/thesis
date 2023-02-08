import json
import os.path

from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import smooth_foveate, one_hot_encoder, realcoord2gridcoord, gridcoord2realcoord, \
    ind2gridcoord
from fixationPrediction.predict import beam_search
from evaluation.accuracyBeatrizC import bbox2cellsbboxcoord
import config

img_path = "/Volumes/DropSave/Tese/dataset/resized_images/clock/000000070285.jpg"
save_dir = "/Users/beatrizpaula/Desktop/Tese/images"

bbox = x, y, w, h = [162, 40, 49, 48]

img = Image.open(img_path)

img.show()

img.save(os.path.join(save_dir, "foveation_original.jpg"))

img_arr = img_to_array(img)

img_50 = smooth_foveate(img_arr, x + w/2, y + h/2, 50)

array_to_img(img_50).save(os.path.join(save_dir, "foveation_50.jpg"))

img_75 = smooth_foveate(img_arr, x + w/2, y + h/2, 75)

array_to_img(img_75).save(os.path.join(save_dir, "foveation_75.jpg"))

img_100 = smooth_foveate(img_arr, x + w/2, y + h/2, 100)

array_to_img(img_100).save(os.path.join(save_dir, "foveation_100.jpg"))

'''path = "/Volumes/DropSave/Tese/dataset/test_pairs_id.json"
img_directory = "/Volumes/DropSave/Tese/dataset/resized_images/"

with open(path) as f:
    test_ids = json.load(f)


for test in test_ids:
    id = test["name"]
    task = test["task"]
    bbox = test["bbox"]
    img_path = img_directory + task + "/" + id

    img = Image.open(img_path)
    img_arr = img_to_array(img)

    # Predict sequence
    task_encoding = one_hot_encoder(task)
    mem_size = config.mem_size
    rnn_path = '/Users/beatrizpaula/Downloads/test5Julbatch128_0.h5'
    rnn = load_model(rnn_path)
    #seqs = beam_search(rnn, img_arr, task_encoding, mem_size, foveate_function=smooth_foveate)


    #fov_array = smooth_foveate(image_array, x, y, fovea_size)
    #fov_image = array_to_img(fov_array)

    #draw = ImageDraw.Draw(fov_image)

    # Open image file
    image = Image.open(img_path)

    font = ImageFont.truetype("arial.ttf", 18)

    # ***** Draw Grid *****
    draw = ImageDraw.Draw(image, "RGBA")
    y_start = 0
    y_end = image.height
    step_size = 32

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill="grey")

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill="grey")

    # ***** Print grid coord *****
    step_size = 32

    for x in range(0, image.width, step_size):
        text = str(x)
        draw.text((x, 0), text, fill="white")

    for y in range(0, image.height, step_size):
        text = str(y)
        draw.text((0, y), text, fill="white")

    # ***** Draw BBOX *****
    shape = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
    draw.rectangle(shape, outline='black', width=2)

    print(shape)
    shape2 = bbox2cellsbboxcoord(bbox)
    shape = (shape2[0][0], shape2[0][1], shape2[1][0], shape2[1][1])
    print(shape)

    draw.rectangle(shape, outline='blue', fill=(256, 256, 256, 127), width=2)

    # ***** Draw Prediction *****
    points = []
    for i in range(7):
        x, y = ind2gridcoord(seqs[0][i])
        x, y = gridcoord2realcoord(x, y)
        points.append((x, y))

    draw.line(points, fill="purple")

    for i in range(7):
        x, y = ind2gridcoord(seqs[0][i])
        x, y = gridcoord2realcoord(x, y)
        shape = (x - 2, y - 2, x + 2, y + 2)
        draw.ellipse(shape, outline="purple", fill="purple", width=5)
        text = str(i)
        w, h = font.getsize(text)
        x = x + 2
        y = y + 2
        draw.rectangle((x, y, x + w, y + h), fill=(149, 165, 166, 127))  # sheer white background
        draw.text((x, y), text, font=font, fill="purple")

    del draw

    image.show()
'''




