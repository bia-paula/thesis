import json
import os.path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
from dataPreprocessing.foveateImages import smooth_foveate, realcoord2gridcoord, gridcoord2realcoord, ind2gridcoord


path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
img_directory = "/Volumes/DropSave/Tese/dataset/resized_images/"
save_dir = "/Users/beatrizpaula/Desktop/Tese/images"

path = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
predicted_model = "/Volumes/DropSave/Tese/trainedModels/fov100_batch256_normal_rnny_onehot_label/testing"

img_id = "000000383549"

with open(path) as f:
    test_ids = json.load(f)

fontsize = -2
font = ImageFont.truetype("arial.ttf", fontsize)


# optionally de-increment to be sure it is less than criteria




#for test in test_ids[:5]:
for test in [test_ids[img_id]]:
    for task in test.keys():
        '''id = test["name"]
        task = test["task"]
        bbox = test["bbox"]'''

        id = img_id
        task = task
        bbox = test_ids[img_id][task]
        img_path = img_directory + task + "/" + id + ".jpg"

        img = Image.open(img_path)
        img_arr = img_to_array(img)

        with np.load(os.path.join(predicted_model, task, id + ".npz"), allow_pickle=True) as data:
            seq = data["seqs"][0]

        #fov_array = smooth_foveate(image_array, x, y, fovea_size)
        #fov_image = array_to_img(fov_array)

        #draw = ImageDraw.Draw(fov_image)

        # Open image file
        image = Image.open(img_path)

        font = ImageFont.truetype("arial.ttf", 16)
        draw = ImageDraw.Draw(image, "RGBA")
        '''# ***** Draw Grid *****
        draw = ImageDraw.Draw(image,"RGBA")
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
            draw.line(line, fill="grey")'''

        '''# ***** Print grid coord *****
        step_size = 32
    
        for x in range(0, image.width, step_size):
            text = str(x)
            draw.text((x, 0), text, fill="white")
    
        for y in range(0, image.height, step_size):
            text = str(y)
            draw.text((0, y), text, fill="white")'''

        # ***** Draw BBOX *****
        shape = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        draw.rectangle(shape, outline='black', width=2)

        # ***** Draw FP Truth (real coord) *****
        '''points = []
        for i in range(test["length"]):
            points.append((test["X"][i], test["Y"][i]))
        draw.line(points, fill="blue")'''
        points = []
        for i in seq:
            grid_x, grid_y = ind2gridcoord(i)
            x, y = gridcoord2realcoord(grid_x, grid_y)
            points.append((x, y))
        draw.line(points, fill="red", width=1)

        '''for i in range(test["length"]):
            shape = (test["X"][i]-2, test["Y"][i]-2, test["X"][i]+2, test["Y"][i]+2)
            draw.ellipse(shape, outline="blue", fill="blue", width=5)
            text = str(i)
            w, h = font.getsize(text)
            x = test["X"][i]+2
            y = test["Y"][i]+2
            draw.rectangle((x, y, x + w, y + h), fill=(236,236,236,127)) # sheer white background
            draw.text((x, y), text, font=font, fill="darkblue")'''
        for t, i in enumerate(seq):
            grid_x, grid_y = ind2gridcoord(i)
            x, y = gridcoord2realcoord(grid_x, grid_y)
            shape = (x-2, y-2, x+2, y+2)
            draw.ellipse(shape, outline="red", fill="red", width=5)
            text = str(t)
            w, h = font.getsize(text)
            x0 = x+2
            y0 = y+2
            draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(236,236,236,150)) # sheer white background
            draw.text((x0, y0), text, font=font, fill="black")

        '''# ***** Draw FP Truth (grid coord) *****
        points = []
        for i in range(test["length"]):
            x, y = realcoord2gridcoord(test["X"][i], test["Y"][i])
            x, y = gridcoord2realcoord(x, y)
            points.append((x, y))
        draw.line(points, fill="purple")

        for i in range(test["length"]):
            x, y = realcoord2gridcoord(test["X"][i], test["Y"][i])
            x, y = gridcoord2realcoord(x, y)
            shape = (x - 2, y - 2, x + 2, y + 2)
            draw.ellipse(shape, outline="purple", fill="purple", width=5)
            text = str(i)
            w, h = font.getsize(text)
            x = x + 2
            y = y + 2
            draw.rectangle((x, y, x + w, y + h), fill=(149, 165, 166, 127))  # sheer white background
            draw.text((x, y), text, font=font, fill="purple")'''

        del draw
        image.save(os.path.join(save_dir, "Fix_example_"+task+".jpg"))
        image.show()
