from PIL import Image
import os, sys
from glob import glob

# Script to Resize all Images from coco-SEARCH18 from 1680x1050 to 512x320 as it was done in their research paper

#path = "/Volumes/DropSave/Tese/dataset/images"
#newpath = "/Volumes/DropSave/Tese/dataset/resized_images"

path = "/Users/beatrizpaula/Downloads/images"
newpath = "/Users/beatrizpaula/Desktop/resized_images"


dirs = os.listdir(path)

categories_path = glob(path + "/*")

i = 0

# Iterate over categories folders
for category in categories_path:
    class_name = category.split("/")[-1]

    # Create category folder in newpath
    if not os.path.exists(newpath + "/" + class_name):
        os.makedirs(newpath + "/" + class_name)

    init_images = os.listdir(category + "/")

    for item in init_images:
        im = Image.open(category + "/" + item)
        f, e = os.path.splitext(item)
        imResize = im.resize((512, 320), Image.Resampling.LANCZOS)
        imResize.save(newpath + "/" + class_name + "/" + f + e, quality=95, subsampling=0)

    i += 1
    print("categories " + str(i) + "/" + str(len(categories_path)) + " DONE")


