#---------------------------------------------------------------------
#CSYE7105 - Final Project - Spring 2022
#Team 2
#Contributors: Pramod Gopal, Pratiksha Patole
#Project Title: Performance Analysis of Multiple GPUs for a Scene Recognition Computer Vision Task
#This code is for preprocessing of images and do a comparison of Process Vs Pool technique
#---------------------------------------------------------------------

from pathlib import Path
import os
from PIL import Image, ImageFilter
import numpy as np
import cv2
import time

data_dir = Path("/home/gopal.p/pramod_gopal_csye7105/final_project/datset_v2/proc_pool_data")
data_folders = [f for f in (data_dir).iterdir() if f.suffix == '']
print(data_folders)


# Preprocessing functions

# resize each image in the dataset
def resize_img(image: np.ndarray, short_size: int, max_size: int) -> np.ndarray:
    # Resize an image such that the short side is equal to short size, unless the larger image size would be larger
    # than max_size, while preserving aspect ratio
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    inter_mode = cv2.INTER_AREA
    if scale > 1:
        inter_mode = cv2.INTER_CUBIC
    resized_image = cv2.resize(
        image, (t_width, t_height), interpolation=inter_mode)
    return resized_image, scale

# Check if any images are transparent in the dataset
def has_transparency(img):
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False


transparent_images = [] # Get images that are transparent
bbox = [] # Get bounding box
# The function to be run with Python multiprocessing
def preprocess_images(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    if has_transparency(img):
        transparent_images.append(img_path)
    else:
        resized_img, scale = resize_img(img_array, 300, 500)
        processed_img = Image.fromarray(resized_img)
        
        processed_img = processed_img.effect_spread(2)
        processed_img = processed_img.filter(filter=ImageFilter.BLUR)
        processed_img = processed_img.getbbox() 
        # Save image to new path after preprocessing


# Run preprocessing without multi-processing

start = time.time()
for folder in data_folders:
    image_folders = [f for f in folder.iterdir()]
    #print("image_folder ==",image_folders)
    for i, image_folder in enumerate(image_folders):
         x = [img_path for img_path in image_folder.iterdir()]
        # print("x ===",x)
         #sys.exit()
         for image in x:
             preprocess_images(image)
stop = time.time()
elapsed = stop-start

print("----------------------------------------------------------------------")
print(' ')
print('Total time elapsed without \
    multi-processing is {} in seconds and {} in minutes'.format(elapsed, elapsed/60))
print(' ')    

# Python Multiprocessing - Process and Pool classes

# Process class

import multiprocessing

start = time.time()
processes = []
for folder in data_folders:
    image_folders = [f for f in folder.iterdir()]
    for i, image_folder in enumerate(image_folders):
        x = [img_path for img_path in image_folder.iterdir()]
        for image in x:
            p = multiprocessing.Process(target=preprocess_images, args=(image,))
            processes.append(p)
            p.start()
    for process in processes:
        process.join()
    #processes[i]= None
stop = time.time()
elapsed = stop-start

print("----------------------------------------------------------------------")
print(' ')
print('Total time elapsed with multiprocessing (Process class)\
    is {} in seconds and {} in minutes'.format(elapsed, elapsed/60))
print(' ')    


# Pool class

start = time.time()
pool = multiprocessing.Pool()

for folder in data_folders:
    image_folders = [f for f in folder.iterdir()]
    for i, image_folder in enumerate(image_folders):    
        x = [img_path for img_path in image_folder.iterdir()]
        pool.map(preprocess_images, x) 
    pool.close()
stop = time.time()
elapsed = stop-start

print("----------------------------------------------------------------------")
print(' ')
print('Total time elapsed with multiprocessing (Pool class) is {} in seconds \
    and {} in minutes'.format(elapsed, elapsed/60))
print(' ')    

