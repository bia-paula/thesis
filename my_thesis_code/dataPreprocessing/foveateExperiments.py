import laplacian_foveation as fv
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import json
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import math
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from time import time

from glob import glob
import tensorflow as tf
import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')
import config
from foveateImages import smooth_foveate
import os
import rnn

def fixated_object(fx, fy, x, y, w, h):
    if x <= fx <= x + w and y <= fy <= y + h:
        return True
    else:
        return False

class BatchSequence(Sequence):
    # val = True for validation
    def __init__(self, k, val):
        if not val:
            #self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/train_scanpaths_fov100_*"))
            #self.arr = np.array(["/Users/beatrizpaula/Downloads/onehot75_batch4.1.npz"])
            self.arr = np.array(["/Users/beatrizpaula/Downloads/valid_exp_scanpaths_30jun.npz"])
        else:
            #self.arr = np.array(glob("/Volumes/DropSave/Tese/dataset/val_scanpaths_fov100_*"))
            self.arr = np.array(["/Users/beatrizpaula/Downloads/valid_exp_scanpaths_30jun.npz"])

    def __len__(self):
        return self.arr.size

    def __getitem__(self, idx):
        path = self.arr[idx]
        return load_batch(path)

def load_batch(path):
    with np.load(path, allow_pickle=True) as data:
        return [data['rnn_x'], data['label_encodings']], data['rnn_y']


def cross_validate(K, model_function, vr, **kwargs):
    histories = []
    start = time()
    for k in range(K):
        X = BatchSequence(k, val=False)
        X_val = BatchSequence(k, val=True)
        n_labels = len(config.classes)
        model = model_function(n_labels)
        histories.append(model.fit(X,
                                   validation_data=X_val,  # feed in the test data for plotting
                                   **kwargs).history)
        model.save('/Users/beatrizpaula/Downloads/' + vr + '_' + str(k) + '.h5')
        # plot_histories(histories, vr+str(k))
        print(f'Version {vr}, model {k}')
        print(f"Accuracy: {histories[k]['val_categorical_accuracy'][-1]}")

    print(f'Train time: {time() - start}')

    return histories

im_path = "../detectron/000000578751.jpg"
#plt.imshow(im_path)
fovea_size = 100
image = Image.open(im_path)
img = plt.imshow(image, interpolation='nearest')
plt.axis('off')
plt.savefig("test1.png", bbox_inches='tight', pad_inches = 0)
plt.show()
image_array = img_to_array(image)
x = 256
y = 160

fov_array = smooth_foveate(image_array, x, y, fovea_size)
fov_image = array_to_img(fov_array)
img = plt.imshow(fov_array, interpolation='nearest')
plt.axis('off')
plt.savefig("test2.png", bbox_inches='tight', pad_inches = 0)
plt.show()





'''

heatmaps_dir = "/Volumes/DropSave/Tese/dataset/taskencoding_heatmap.npz"
heatmaps = np.load(heatmaps_dir)["label_heatmap"]

new = np.split(heatmaps, 16, axis=1)
one_task = heatmaps[0]
one_task_new = np.split(heatmaps[0], 10)
one_task_new_arr = np.array(one_task_new)

#one_task_new_arr = np.zeros((10, 16))

#for idx, a in enumerate(one_task_new):
    #one_task_new_arr[idx, :] = one_task_new[idx]

broadcasted_b = np.broadcast_to(one_task_new_arr, (512, 10, 16))
broadcasted = np.moveaxis(broadcasted_b, 0, 2)

for i in range(512):
    if not np.array_equal(one_task_new_arr, broadcasted[:,:,i]):
        print("FALSE: ", i)

confirm = np.zeros((10, 16))
for id in range(160):
    x, y = ind2gridcoord(id)
    confirm[y, x] = one_task[id]

print(np.array_equal(confirm, one_task_new_arr))






path = "/Users/beatrizpaula/Desktop/Tese/my_thesis_code/fixationPrediction/fov100_batch64_normal_rnny_onehot_label_histories.npz"

data = np.load(path, allow_pickle=True)
h = data["histories"]
dh = h.item()

path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"

with open(path) as fp:
    data = json.load(fp)

count = 0
total_fixated = 0
fixated_in_last = 0
for obs in data:
    last_fixated = -1
    for fp_idx in range(min(obs["length"], 7)):
        x, y, w, h = obs["bbox"]
        if fixated_object(obs["X"][fp_idx], obs["Y"][fp_idx], x, y, w, h):
            last_fixated = fp_idx
    if last_fixated > -1:
        total_fixated += 1
        if last_fixated == min(obs["length"]-1, 6):
            fixated_in_last += 1

    count += 1
    if count % 10 == 0:
        print(count, "/", len(data))

print(fixated_in_last)
print(total_fixated)
print(count)



path = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_test.json"
new_file = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
with open(path) as file:
    test_data = json.load(file)

testing_dict = dict()

count = 0

for exp in test_data:
    name = exp["name"].split(".")[0]
    task = exp["task"]
    if not name in testing_dict:
        new_bbox = exp["bbox"]
        new_bbox[0] = int(new_bbox[0] * 512/1680)
        new_bbox[1] = int(new_bbox[1] * 320 / 1050)
        new_bbox[2] = int(new_bbox[2] * 512 / 1680)
        new_bbox[3] = int(new_bbox[3] * 320 / 1050)
        testing_dict[name] = {task: new_bbox}
        count += 1
    elif not task in testing_dict[name]:
        new_bbox = exp["bbox"]
        new_bbox[0] = int(new_bbox[0] * 512 / 1680)
        new_bbox[1] = int(new_bbox[1] * 320 / 1050)
        new_bbox[2] = int(new_bbox[2] * 512 / 1680)
        new_bbox[3] = int(new_bbox[3] * 320 / 1050)
        testing_dict[name][task] = new_bbox
        count += 1

with open(new_file, 'w') as f:
    json.dump(testing_dict, f)

print(count)


testing_pairs = list()

data_dict = {}

testing_dicts =  []

for exp in test_data:
    pair = list()
    pair.append(exp["name"])
    pair.append(exp["task"])
    #pair.append(exp["bbox"])
    testing_pairs.append(pair)
    if not exp["name"] in data_dict:
        data_dict[exp["name"]] = [exp["task"]]
        new_bbox = exp["bbox"]
        new_bbox[0] = int(new_bbox[0] * 512/1680)
        new_bbox[1] = int(new_bbox[1] * 320 / 1050)
        new_bbox[2] = int(new_bbox[2] * 512 / 1680)
        new_bbox[3] = int(new_bbox[3] * 320 / 1050)

        new = {"name": exp["name"], "task": exp["task"], "bbox": new_bbox}
        testing_dicts.append(new)
    else:
        if not exp["task"] in  data_dict[exp["name"]]:
            data_dict[exp["name"]].append(exp["task"])
            new_bbox = exp["bbox"]
            new_bbox[0] = int(new_bbox[0] * 512 / 1680)
            new_bbox[1] = int(new_bbox[1] * 320 / 1050)
            new_bbox[2] = int(new_bbox[2] * 512 / 1680)
            new_bbox[3] = int(new_bbox[3] * 320 / 1050)

            new = {"name": exp["name"], "task": exp["task"], "bbox": new_bbox}
            testing_dicts.append(new)
            

testing_set = set(tuple(x) for x in testing_pairs)
sorted_by_image = sorted(list(testing_set), key=lambda tup: tup[0])
#testing_set = set(testing_pairs)

new_file = "/Volumes/DropSave/Tese/dataset/test_pairs_id.json"

with open(new_file, 'w') as f:
    json.dump(testing_dicts, f)

count = 0

for img in data_dict.keys():
    if len(data_dict[img]) > 1:
        print(img)
        print(data_dict[img])
        count+=1
print(count)
print(len(testing_dicts))
print(len(testing_set))

h1 = np.load("/Users/beatrizpaula/Desktop/Tese/my_thesis_code/fixationPrediction/test5Julbatch128histories1.npz", allow_pickle=True)
h2 = np.load("/Users/beatrizpaula/Desktop/Tese/my_thesis_code/fixationPrediction/test5Julbatch128histories2.npz", allow_pickle=True)

data1 = h1["histories"]
data2 = h2["arr_0"]

model1 = data1[0]

for i in range(5):
    for j in range(10):
        if lens[f] >= 380 
            start = 3800 * i + 380 * j
            finish = start + 380
            save_X = X[start:finish]
        print(save_X.shape)
    print("Batch ", i, " done!")
first = np.zeros((10,7,10,16))
second = np.zeros((15,7,10,16))

merge = np.append(first, second, axis=0)
'''
'''
x = range(7,7)
y = range(6,7)
z = range(8,7)
for i in range(7,7):
    print(i)
print("aqui")
for i in range(6,7):
    print(i)

mine = BatchSequence(0, val=True)[0]
mine_in = mine[0]
mine_out = mine[1]
mine_in_images = mine_in[0]
mine_in_task = mine_in[1]
m = np.Inf
for seq in mine_in_images:
    #print(seq.shape)
    #print(type(seq))
    print(seq.min())
    if m < seq.min():
        m = seq.min()

print(m)

#print(mine_in_images)

padded_in = tf.keras.preprocessing.sequence.pad_sequences(mine_in_images, maxlen=7, dtype="float64", padding='post',truncating='post', value=-1)

print("AFTER PADDING")
for seq in padded_in:
    print(seq.shape)
#print(padded_in)

view = padded_in[0][0]
view_2 = padded_in[0][3]

tf_xtrain= tf.convert_to_tensor (padded_in,dtype='float64')
masking_layer = tf.keras.layers.Masking(mask_value=-1)
masked_x_train = masking_layer(tf_xtrain)

print(masked_x_train._keras_mask[0][3])


model_f = rnn.create_model
vr="first_try"
cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
histories = cross_validate(K=5, model_function=model_f, vr=vr, epochs=5, callbacks=[cb])


theirs = BatchSequence(0, val=False)
print("1")
mine = BatchSequence(0, val=True)
print("2")
n_labels = theirs[0]
n_labels_0 = theirs[0][0]
n_labels_1 = theirs[0][1]
n_labels_00 = theirs[0][0][0]
n_labels_01 = theirs[0][0][1]
n_labels_mine = mine[0]
n_labels_mine_0 = mine[0][0]
n_labels_mine_1 = mine[0][1]
n_labels_mine_00 = mine[0][0][0]
n_labels_mine_01 = mine[0][0][1]'''

#path_fp_train = "/Users/beatrizpaula/Downloads/exp_scanpaths_23jun.npz"
#path_fp_val = "/Users/beatrizpaula/Downloads/valid_exp_scanpaths_23jun.npz"





'''
path_fp_train = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json"
fp = open(path_fp_train)
training_data = json.load(fp)  # list of dictionaries with each experiment

count1 = len(training_data)
print(count1)

path_fp_val = "/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_valid.json"
fp = open(path_fp_val)
val_data = json.load(fp)  # list of dictionaries with each experiment


not_in_six = []
needed_fix = []
lengthmorethansix = 0
large_seq_but_fixed_before = []
j = 0

for exp in training_data:
    f = False
    for i in range(exp["length"]):
        if fixated_object(exp["X"][i], exp["Y"][i], exp["bbox"][0], exp["bbox"][1], exp["bbox"][2], exp["bbox"][3]):
            f = True
            needed_fix.append(i)
            break

    if i > 7:
        not_in_six.append(exp)

    if exp["length"] > 7:
        lengthmorethansix+=1

    if exp["length"] > 7 and i<=7:
        large_seq_but_fixed_before.append(exp)


    #j += 1
    #print(str(j) + "/" + str(len(train)) + " DONE")

needed_fix_val = []
for exp in val_data:
    f = False
    for i in range(exp["length"]):
        if fixated_object(exp["X"][i], exp["Y"][i], exp["bbox"][0], exp["bbox"][1], exp["bbox"][2], exp["bbox"][3]):
            f = True
            needed_fix_val.append(i)
            break

print(min(needed_fix_val))
print(max(needed_fix_val))

print(min(needed_fix))
print(max(needed_fix))

needed_fix_np = np.array(needed_fix)
print(np.mean(needed_fix_np))
print(np.median(needed_fix_np))

needed_fix_val_np = np.array(needed_fix_val)
print(np.mean(needed_fix_val_np))
print(np.median(needed_fix_val_np))


plt.hist(needed_fix_val_np, 37)
plt.show()

#higher than 6
print("Higher than 6: " + str(len(needed_fix_np[needed_fix_np>6])))
print("Higher than 6: " + str(len(needed_fix_val_np[needed_fix_val_np>6])))

#higher than 7
print("Higher than 7: " + str(len(needed_fix_np[needed_fix_np>7])))
print("Higher than 7: " + str(len(needed_fix_val_np[needed_fix_val_np>7])))

#higher than 8
print("Higher than 8: " + str(len(needed_fix_np[needed_fix_np>8])))
print("Higher than 8: " + str(len(needed_fix_val_np[needed_fix_val_np>8])))

#higher than 9
print("Higher than 9: " + str(len(needed_fix_np[needed_fix_np>9])))
print("Higher than 9: " + str(len(needed_fix_val_np[needed_fix_val_np>9])))

count1 = len(training_data)
count2 = len(val_data)
print(count1+count2)


f = np.load('/Volumes/DropSave/Tese/dataset/train_scanpaths_fov100_0.npz', allow_pickle=True)

rnn_x = f['rnn_x'][0]
rnn_y = f['rnn_y'][0]
label_encoding = f['label_encodings'][0]

print(rnn_x.shape)

for i in rnn_x[:10]:
    print(i.shape)

print(rnn_x.shape)

# plt.imshow(im_f)
fovea_size = 100
image = Image.open('/Volumes/DropSave/Tese/dataset/resized_images/bottle/000000001455.jpg')
image_array = img_to_array(image)
x = 275.20000000000005
y = 59.73333333333334

fov_array = smooth_foveate(image_array, x, y, fovea_size)
fov_image = array_to_img(fov_array)

draw = ImageDraw.Draw(fov_image)
draw.rectangle([274, 5, 274 + 32, 5 + 74], outline="black")
fov_image.show()
# fov_image.show()


f = open("/Volumes/DropSave/Tese/dataset/human_scanpaths_TP_trainval_train.json")
train = json.load(f)
max_l = 0
l = []

for exp in train:
    l.append(exp['length'])
    if exp['length'] > max_l:
        max_l = exp['length']
        ind = exp

#print(max_l)
#print(ind)
#print(min(l))
#print(max(l))

#len_np = np.array(l)
#print(np.mean(len_np))
#print(np.median(len_np))

#plt.hist(len_np, 41)
#plt.show()

not_in_six = []
needed_fix = []
j = 0
for exp in train:
    f = False
    for i in range(exp["length"]):
        if fixated_object(exp["X"][i], exp["Y"][i], exp["bbox"][0], exp["bbox"][1], exp["bbox"][2], exp["bbox"][3]):
            f = True
            needed_fix.append(i)
            break

    if i > 6:
        not_in_six.append(exp)

    #j += 1
    #print(str(j) + "/" + str(len(train)) + " DONE")

print(min(needed_fix))
print(max(needed_fix))

needed_fix_np = np.array(needed_fix)
print(np.mean(needed_fix_np))
print(np.median(needed_fix_np))

plt.hist(needed_fix_np, 37)
plt.show()

#higher than 6
print("Higher than 6: " + str(len(needed_fix_np[needed_fix_np>6])))

#higher than 7
print("Higher than 7: " + str(len(needed_fix_np[needed_fix_np>7])))

#higher than 8
print("Higher than 8: " + str(len(needed_fix_np[needed_fix_np>8])))

#higher than 9
print("Higher than 9: " + str(len(needed_fix_np[needed_fix_np>9])))

m = np.mean(needed_fix_np)
s = np.std(needed_fix_np)

print(m)
print(s)
print(m+s)
print(m+2*s)
print(m+3*s)

newpath = "/Volumes/DropSave/Tese/dataset/resized_images"
init_images = os.listdir(newpath + "/")
print(init_images[1:])

cnn = VGG16(weights='imagenet', include_top=False, input_shape=(405, 405, 3))
cnn.summary()'''

''' # Iterate over experiments
 for obs in training_data:
     # Get image
     class_name = obs["task"]
     image_id = obs["name"]
     image_path = image_dir + "/" + class_name + "/" + image_id
     img = Image.open(image_path)
     img_array = img_to_array(img)

     # holder for features of foveated images
     fix = np.empty((0, fmap_size[0], fmap_size[1], fmap_size[2]))

     # iterate over fp
     for i_fp in range(obs["length"]):
         x = obs["X"][i_fp]
         y = obs["Y"][i_fp]

         # Foveate image on current fp
         fov_array = smooth_foveate(img_array, x, y, fovea_size)
         #fov_image = array_to_img(fov_array)
         #fov_image.show()

         # Add fixation features to array
         fmap = get_fmap(fov_array, fmap_model)
         fix = np.append(fix, fmap, axis=0)

     rnn_x.append(fix)
     count+=1
     print(str(count) + "/" + str(len(training_data)))

     label_encodings.append(one_hot_encoder(class_name))

     if count == 10: break


 for item in label_encodings:
     print(item.shape)'''