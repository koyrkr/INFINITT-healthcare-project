import numpy as np
import cv2, time
import matplotlib.pyplot as plt

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from skimage.transform import resize
from PIL import Image, ImageEnhance

import os, glob

json_file = open("model.json", "r")

loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")


pred_list = sorted(glob.glob('CHNCXR_png/*.png'))
pred_list = pred_list[0:50]
new_pred_list = []
t = time.process_time()
count = 0
for item in pred_list:
    enhancer_ct = ImageEnhance.Contrast(Image.open(item).convert('RGB'))
    enhancer_ct.enhance(1.75).save('temp/temp_' + str(count) + '.png')
    new_pred_list.append('temp/temp_' + str(count) + '.png')
    count += 1

IMG_SIZE = 512

pred_data, dum_data = np.empty((2, len(new_pred_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
height_list = []
width_list = []

for i, pred_path in enumerate(new_pred_list):
    img = imread(pred_path)
    height, width, depth = img.shape
    height_list.append(height)
    width_list.append(width)
    print(str(height) + ' ' + str(width))
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    pred_data[i] = img;

preds = loaded_model.predict(pred_data)

fig, ax = plt.subplots(10, 2, figsize=(10, 100))

count = 0
for i, pred in enumerate(preds):
    imsave('temp/res_' + str(count) + '.png', pred.squeeze());
    dst = cv2.fastNlMeansDenoising(imread('temp/res_' + str(count) + '.png'), None, 30.0, 7, 21);
    dst = resize(dst, output_shape=(height_list[count], width_list[count]), preserve_range=True)
    imsave('temp/res_' + str(count) + '.png', dst);
    start_idx = 0
    end_idx = 0
    cnt = 0
    for j in range(0, len(dst)):
        flag = False
        for k in range(0, len(dst[0])):
            if dst[j][k] >= 220:
                flag = True
                start_idx = j - cnt if start_idx == 0 and cnt >= 200 else start_idx
                end_idx = j if cnt >= 200 else end_idx
        cnt = cnt + 1 if flag else 0
    print(start_idx)
    print(end_idx)
    for j in range(0, len(dst[0])):
        for k in range(0, 15):
            dst[start_idx+k-7][j] = 255
            dst[end_idx+k-7][j] = 255

    ax[count % 10, 0].imshow(pred_data[i].squeeze(), cmap='gray')
    ax[count % 10, 1].imshow(dst, cmap='gray')
    count += 1
    if count % 10 == 0:
        plt.savefig('pred_image/pred_image_' + str(count / 10 - 1) + '.png')
print((time.process_time()-t)/count)
