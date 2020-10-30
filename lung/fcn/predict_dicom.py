import numpy as np
import cv2, time, pydicom, sys, math
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from PIL import Image, ImageEnhance
from collections import deque

import os, glob

IMG_SIZE = 512



print("\n\n----- load model -----")

json_file = open("200529_model.json", "r")

loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("200529_model.h5")



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
error_list = sorted(glob.glob('error_image/*.jpg'))
error_list = [item.split("/")[1] for item in error_list]

pneumo_list = sorted(glob.glob('../dataset/siim/input_pneumo_dicom/*.dcm'))
pneumo_list = [item.split("/")[4] for item in pneumo_list]
normal_list = sorted(glob.glob('../dataset/siim/input_normal_dicom/*.dcm'))
normal_list = [item.split("/")[4] for item in normal_list]

tmp_pred_list = []
tmp_test_answer = []
pixel_spacing_list = []

for n, item in enumerate(tqdm(pred_list)):
    for i, fn in enumerate(test_list):
        # if item.split("/")[4] in pneumo_list:
        #     continue
        if fn == item.split("/")[4].replace('.dcm', '.jpg'):
            tmp_test_answer.append(test_answer[i])
            tmp_pred_list.append(item)
            break

t = time.thread_time()

pred_data = np.empty((1, len(tmp_pred_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)[0]
height_list = []
width_list = []
pixel_mean_list = []

for i, pred_path in enumerate(tqdm(tmp_pred_list)):
    dcmfile = pydicom.dcmread(pred_path)
    pixel_array = dcmfile.pixel_array
    # pixel_array = cv2.equalizeHist(pixel_array)
    height_list.append(len(pixel_array))
    width_list.append(len(pixel_array[0]))
    pixel_spacing_list.append(dcmfile.PixelSpacing)
    img = resize(pixel_array, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    pred_data[i] = img;

print("\n\n----- predict test case -----")
preds = loaded_model.predict(pred_data)



print("\n\n----- make result -----")

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
top_cor_list = []
bot_cor_list = []
for cnt, pred in enumerate(tqdm(preds)):
    # dst = cv2.fastNlMeansDenoising(img_as_ubyte(pred.squeeze()), None, 30.0, 7, 21);
    dst = resize(img_as_ubyte(pred.squeeze()), output_shape=(height_list[cnt], width_list[cnt]), preserve_range=True)
    dst = np.array([[dst[i][j] for j in range(0, len(dst[i]), 16)] for i in range(0, len(dst), 16)])
    start_idx = -1
    end_idx = len(dst)


    # BFS code for finding largest two clusters
    group_table = dict()
    map = [[0 if dst[i][j] <= 200 else -1 for j in range(0, len(dst[i]))] for i in range(0, len(dst))]
    group_num = 0
    total_num = -sum([sum(map[i]) for i in range(len(map))])
    checked_num = 0

    while checked_num < total_num:
        group_num += 1
        x, y = (-1, -1)
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == -1:
                    x, y = (i, j)
                    break
            if x != -1 and y != -1:
                break
        queue = deque()
        queue.append((x, y))
        while queue:
            item = queue.popleft()
            if map[item[0]][item[1]] == -1:
                map[item[0]][item[1]] = group_num
                checked_num += 1
                group_table[group_num] = 0 if group_num not in group_table else group_table[group_num] + 1
                if item[0] > 0 and map[item[0]-1][item[1]] == -1:
                    queue.append((item[0]-1, item[1]))
                if item[0] < len(map) - 1 and map[item[0]+1][item[1]] == -1:
                    queue.append((item[0]+1, item[1]))
                if item[1] > 0 and map[item[0]][item[1]-1] == -1:
                    queue.append((item[0], item[1]-1))
                if item[1] < len(map[0]) - 1 and map[item[0]][item[1]+1] == -1:
                    queue.append((item[0], item[1]+1))
    sorted_group_table = sorted(group_table.items(), key=(lambda x: x[1]), reverse=True)
    it = iter(sorted_group_table)
    start_idx_1, end_idx_1 = (len(map), -1)
    start_idx_2, end_idx_2 = (len(map), -1)
    fst_group = next(it, None)
    if fst_group:
        for i, j in zip(range(len(map)), reversed(range(len(map)))):
            if start_idx_1 == len(map) and fst_group[0] in map[i]:
                start_idx_1 = i
            if end_idx_1 == -1 and fst_group[0] in map[j]:
                end_idx_1 = j
            if start_idx_1 != len(map) and end_idx_1 != -1:
                break
    snd_group = next(it, None)
    if snd_group:
        for i, j in zip(range(len(map)), reversed(range(len(map)))):
            if start_idx_2 == len(map) and snd_group[0] in map[i]:
                start_idx_2 = i
            if end_idx_2 == -1 and snd_group[0] in map[j]:
                end_idx_2 = j
            if start_idx_2 != len(map) and end_idx_2 != -1:
                break

    start_idx = min(start_idx_1, start_idx_2) * 16
    end_idx = max(end_idx_1, end_idx_2) * 16

    top_cor_list.append((start_idx - tmp_test_answer[cnt][0]) * pixel_spacing_list[cnt][0])
    bot_cor_list.append((end_idx - tmp_test_answer[cnt][1]) * pixel_spacing_list[cnt][1])


print("\n\n----- save test result -----")

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
print(time.thread_time())
print(round((time.thread_time()-t)/len(tmp_test_answer), 3), "sec/image")
