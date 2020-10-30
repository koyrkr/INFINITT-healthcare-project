import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

import os, glob, csv, sys

img_list = sorted(glob.glob('../dataset/CXR/CXR_png/*.png'))
mask_list = sorted(glob.glob('../dataset/CXR/masks/*.png'))

new_img_list = []
for item in mask_list:
    for item2 in img_list:
        if item[21:34] + '.png' in item2:
            new_img_list.append(item2)

img_list = new_img_list

print(len(img_list), len(mask_list))

total_mean_list = []
mask_mean_list = []

for img1, img2 in zip(tqdm(img_list), tqdm(mask_list)):
    pixel_array1 = imread(img1)
    pixel_array1 = resize(pixel_array1, (512, 512), preserve_range=True)
    sum1 = sum([sum(row) for row in pixel_array1])
    total_mean_list.append(sum1 / (512 * 512))
    pixel_array2 = imread(img2)
    pixel_array2 = resize(pixel_array2, (512, 512), preserve_range=True)
    sum2 = sum([sum([pixel_array1[i][j] for j in range(len(pixel_array1[i])) if pixel_array2[i][j] >= 250]) for i in range(len(pixel_array1))])
    cnt = sum([sum([1 for j in range(len(pixel_array2[i])) if pixel_array2[i][j] >= 250]) for i in range(len(pixel_array2))])
    mask_mean_list.append(sum2 / cnt)


w = open('mean_list.csv', 'w', newline='')
wr = csv.writer(w)
wr.writerow(['total', 'mask'])
for i, j in zip(total_mean_list, mask_mean_list):
    wr.writerow([i, j])
w.close()
