from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import pydicom

import os, glob

from tqdm import tqdm

IMG_SIZE = 512

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

tmp_pred_list = []

for n, item in enumerate(tqdm(pred_list)):
    for i, fn in enumerate(test_list):
        if fn == item.split("/")[4].replace('.dcm', '.jpg'):
            tmp_pred_list.append(item)
            break

val_list = []
for i, pred_path in enumerate(tqdm(tmp_pred_list)):
    dcmfile = pydicom.dcmread(pred_path)
    pixel_array = dcmfile.pixel_array
    # height_list.append(len(pixel_array))
    # width_list.append(len(pixel_array[0]))
    # pixel_spacing_list.append(dcmfile.PixelSpacing)
    img = resize(pixel_array, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    pixel_mean = sum([sum(x) for x in img]) / (len(img) * len(img[0]))
    if type(pixel_mean) == np.ndarray:
        pixel_mean = pixel_mean[0]
    # min_val, max_val = max(0, int(pixel_mean - 20 - 40)), min(255, int(pixel_mean - 20 + 40))
    # LUT = np.zeros(256, dtype=np.uint8)
    # LUT[min_val:max_val+1]=np.linspace(start=50,stop=205,num=(max_val-min_val)+1,endpoint=True,dtype=np.uint8)
    # LUT[:min_val]=np.linspace(start=0,stop=50,num=min_val,endpoint=True,dtype=np.uint8)
    # LUT[max_val+1:]=np.linspace(start=205,stop=255,num=(255-max_val),endpoint=True,dtype=np.uint8)
    # LUT = 255. / (1 + np.exp(-(LUT/25.5 - 5)))
    val_list.append(pixel_mean)
    if pixel_mean < 90 or pixel_mean > 150:
        imsave('error_image/' + pred_path.split("/")[4].replace('.dcm', '.jpg'), pixel_array)
    # img = LUT[img.astype(np.uint8)]
print(sum(val_list) / len(val_list))
