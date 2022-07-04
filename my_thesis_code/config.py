original_image_size = (512, 320) # (405, 405)
fmap_size = (10, 16, 512)
image_array_shape = (320, 512, 3)
cnn_output_shape = (10, 16, 512)
classes = ['bottle', 'bowl', 'car', 'chair', 'clock',
           'cup', 'fork', 'keyboard', 'knife', 'laptop',
           'microwave', 'mouse', 'oven', 'potted plant', 'sink',
           'stop sign', 'toilet', 'tv']
fovea_size = 100

memory_size = 20


'''classes = ["car", "cat", "tvmonitor", "chair", "boat", "pottedplant", "bottle", "motorbike"]
labels = ["car", "cat", "tvmonitor", "chair", "boat", "pottedplant", "bottle", "motorbike", "person"]
max_fixations = 9
memory_size = 20
sd_original_size = 405
fmap_size = 12
exp_path = '../../../Experimental/experiment/exp_data_'
dataset_path = '../../datasets/filtered_complete_'
cnn_output_shape = (12, 12, 512)
loc_grid_size = cnn_output_shape[:1]'''