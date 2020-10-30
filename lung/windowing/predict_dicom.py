import numpy as np
import cv2, time, pydicom, sys, math, copy
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from PIL import Image, ImageEnhance
from collections import deque

import os, glob

IMG_SIZE = 64


print("\n\n----- make test answer -----")

test_list = []
test_answer = []
csv_file = open("../dataset/siim_box_coord.csv", "r")
test_flag = False
for row in csv_file:
    if not test_flag:
        test_flag = True
        continue
    if row.split(",")[5] == "\"{}\"":
        continue
    test_list.append(row.split(",")[0])
    y = int(row.split(",")[7].split(":")[1])
    h = int(row.split(",")[9].split(":")[1].split("}")[0])
    test_answer.append((y, y+h))



print("\n\n----- make test image -----")

pred_list = sorted(glob.glob('../dataset/siim/input_all_dicom/*.dcm'))

pneumo_list = sorted(glob.glob('../dataset/siim/input_pneumo_dicom/*.dcm'))
pneumo_list = [item.split("/")[4] for item in pneumo_list]
normal_list = sorted(glob.glob('../dataset/siim/input_normal_dicom/*.dcm'))
normal_list = [item.split("/")[4] for item in normal_list]

tmp_pred_list = []
tmp_test_answer = []

for n, item in enumerate(tqdm(pred_list)):
    for i, fn in enumerate(test_list):
        # if item.split("/")[4] in normal_list:
        #     continue
        if fn == item.split("/")[4].replace('.dcm', '.jpg'):
            tmp_test_answer.append(test_answer[i])
            tmp_pred_list.append(item)
            break

t = time.thread_time()

top_cor_list = []
bot_cor_list = []

for cnt, pred_path in enumerate(tqdm(tmp_pred_list)):
    dcmfile = pydicom.dcmread(pred_path)
    pixel_array = dcmfile.pixel_array
    h, w = pixel_array.shape
    pixel_spacing = dcmfile.PixelSpacing

    pixel_array = copy.deepcopy(pixel_array)

    pixel_array = cv2.equalizeHist(pixel_array)
    pixel_array = resize(pixel_array, (IMG_SIZE, IMG_SIZE), preserve_range=True)

    pixel_mean = int(sum([sum(row) for row in pixel_array]) / (h * w))
    pixel_mean = 75 if pixel_mean < 140 else pixel_mean - 16

    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[i])):
            if pixel_array[i][j] <= pixel_mean:
                pixel_array[i][j] = 255
            else:
                break
        for j in reversed(range(len(pixel_array[i]))):
            if pixel_array[i][j] <= pixel_mean:
                pixel_array[i][j] = 255
            else:
                break

    for i in range(len(pixel_array[0])):
        for j in range(len(pixel_array)):
            if pixel_array[j][i] <= pixel_mean or pixel_array[j][i] == 255:
                pixel_array[j][i] = 255
            else:
                break
        for j in reversed(range(len(pixel_array))):
            if pixel_array[j][i] <= pixel_mean or pixel_array[j][i] == 255:
                pixel_array[j][i] = 255
            else:
                break

    pixel_array = pixel_array.astype(np.uint8)

    lut_min = np.linspace(255, 255, num=(pixel_mean-40), endpoint=True)
    lut_mid = np.linspace(0, 0, num=80, endpoint=True)
    lut_max = np.linspace(255, 255, num=256-(pixel_mean+20), endpoint=True)

    lut = np.concatenate((lut_min, lut_mid, lut_max))
    pixel_array = lut[pixel_array]
    pixel_array = [[pixel_array[i][j] for j in range(0, len(pixel_array[i]))] for i in range(0, len(pixel_array))]

    xs = [x for x in range(len(pixel_array))]
    y = [sum(row)/len(pixel_array[0]) for row in pixel_array]

    start_idx = -1
    end_idx = len(pixel_array)
    for i, x in enumerate(y):
        if x <= 180 and start_idx == -1:
            start_idx = i
        if x <= 220:
            end_idx = i

    start_idx = int(start_idx * h / IMG_SIZE)
    end_idx = int(end_idx * h / IMG_SIZE)

    # tmptmptmptmp
    # img = np.stack((img,)*3, axis=-1)
    # for j in range(0, len(img[0])):
    #     for k in range(0, 15):
    #         if start_idx+k-7 >= 0 and start_idx+k-7 < len(img):
    #             img[start_idx+k-7][j] = [255, 255, 0]
    #         if end_idx+k-7 >= 0 and end_idx+k-7 < len(img):
    #             img[end_idx+k-7][j] = [255, 255, 0]

    top_cor_list.append((start_idx - tmp_test_answer[cnt][0]) * pixel_spacing[0])
    bot_cor_list.append((end_idx - tmp_test_answer[cnt][1]) * pixel_spacing[1])


print("\n\n----- save test result -----")
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].set_ylim(-30, 30)
ax[1].set_ylim(-30, 30)
ax[0].set_ylabel("predict value - answer(mm)")
ax[0].set_xlabel("image index")
ax[1].set_ylabel("predict value - answer(mm)")
ax[1].set_xlabel("image index")
ax[0].set_title("top coordinate")
ax[1].set_title("bottom coordinate")

w = open('top_cor_list.txt', 'w', encoding='utf-8')
for i, x in enumerate(top_cor_list):
    w.write(str(round(x, 3)) + '\n')
w.close()

top_train, top_val = train_test_split(top_cor_list, test_size = 0.3)
top_train_mean = sum(top_train) / len(top_train)
top_val = [x - top_train_mean for x in top_val]

print("top coord mean of deviation:", round(top_train_mean, 3), "(mm)")
top_sd = round(math.sqrt(sum([(x - top_train_mean) * (x - top_train_mean) for x in top_train]) / (len(top_train) - 1)), 3)
print("top coord standard deviation:", top_sd, "(mm)")
ax[0].axhline(y=0, color='r', linewidth=1)
ax[0].axhline(y=-15, color='r', linewidth=0.25)
ax[0].axhline(y=15, color='r', linewidth=0.25)
ax[0].axhline(y=-10, color='r', linewidth=0.5)
ax[0].axhline(y=10, color='r', linewidth=0.5)
ax[0].axhline(y=-5, color='r', linewidth=0.75)
ax[0].axhline(y=5, color='r', linewidth=0.75)

top_cor_list_in = [x for x in top_val if x >= -15 and x <= 15]
print("top coord accuracy(within 15mm): ", round(len(top_cor_list_in)/len(top_val), 3))

top_cor_list_in = [x for x in top_val if x >= -10 and x <= 10]
print("top coord accuracy(within 10mm): ", round(len(top_cor_list_in)/len(top_val), 3))

top_cor_list_in = [x for x in top_val if x >= -5 and x <= 5]
print("top coord accuracy(within 5mm): ", round(len(top_cor_list_in)/len(top_val), 3))


w = open('bot_cor_list.txt', 'w', encoding='utf-8')
for x in bot_cor_list:
    w.write(str(round(x, 3)) + '\n')
w.close()

bot_train, bot_val = train_test_split(bot_cor_list, test_size = 0.3)
bot_train_mean = sum(bot_train) / len(bot_train)
bot_val = [x - bot_train_mean for x in bot_val]

print("bot coord mean of deviation:", round(bot_train_mean, 3), "(mm)")
bot_sd = round(math.sqrt(sum([(x - bot_train_mean) * (x - bot_train_mean) for x in bot_train]) / (len(bot_train) - 1)), 3)
print("bot coord standard deviation:", bot_sd, "(mm)")
ax[1].axhline(y=0, color='r', linewidth=1)
ax[1].axhline(y=-15, color='r', linewidth=0.25)
ax[1].axhline(y=15, color='r', linewidth=0.25)
ax[1].axhline(y=-10, color='r', linewidth=0.5)
ax[1].axhline(y=10, color='r', linewidth=0.5)
ax[1].axhline(y=-5, color='r', linewidth=0.75)
ax[1].axhline(y=5, color='r', linewidth=0.75)

bot_cor_list_in = [x for x in bot_val if x >= -15 and x <= 15]
print("bot coord accuracy(within 15mm): ", round(len(bot_cor_list_in)/len(bot_val), 3))

bot_cor_list_in = [x for x in bot_val if x >= -10 and x <= 10]
print("bot coord accuracy(within 10mm): ", round(len(bot_cor_list_in)/len(bot_val), 3))

bot_cor_list_in = [x for x in bot_val if x >= -5 and x <= 5]
print("bot coord accuracy(within 5mm): ", round(len(bot_cor_list_in)/len(bot_val), 3))

xs = [x for x in range(len(top_val))]
ax[0].plot(xs, top_val, '.', markersize=1.2)
ax[1].plot(xs, bot_val, '.', markersize=1.2)

w = open('lung_size_list.txt', 'w', encoding='utf-8')
for x, y in zip(top_cor_list, bot_cor_list):
    w.write(str(round(x - y, 3)) + '\n')
w.close()

plt.savefig('result.png')


print("\n\n----- print running time -----")

print(round((time.thread_time()-t)/len(tmp_test_answer), 3), "sec/image")
