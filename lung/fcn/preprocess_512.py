import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

import os, glob

img_list = sorted(glob.glob('../dataset/CXR/CXR_png/*.png'))
mask_list = sorted(glob.glob('../dataset/CXR/masks/*.png'))

new_img_list = []
for item in mask_list:
    for item2 in img_list:
        if item[21:34] + '.png' in item2:
            new_img_list.append(item2)

img_list = new_img_list

print(len(img_list), len(mask_list))

IMG_SIZE = 512

x_data, y_data = np.empty((2, len(img_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# save train dataset resized
for i, img_path in enumerate(img_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    x_data[i] = img

# save mask dataset resized
for i, img_path in enumerate(mask_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    y_data[i] = img

y_data /= 255.

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_data[12].squeeze(), cmap='gray')
ax[1].imshow(y_data[12].squeeze(), cmap='gray')

# split dataset into train, validation and test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# save numpy array dataset
np.save('../dataset/CXR/data/x_train.npy', x_train)
np.save('../dataset/CXR/data/y_train.npy', y_train)
np.save('../dataset/CXR/data/x_val.npy', x_val)
np.save('../dataset/CXR/data/y_val.npy', y_val)
np.save('../dataset/CXR/data/x_test.npy', x_test)
np.save('../dataset/CXR/data/y_test.npy', y_test)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
