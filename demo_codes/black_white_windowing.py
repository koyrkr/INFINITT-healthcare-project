'''
    def black_white_windowing:
        들어온 DICOM 파일을 gray scale 형식의 이미지로 리턴하는 메소드

    def coord_windowing:
        들어온 DICOM 파일을 windowing 방식으로 Landmark의 위/아래 좌표를 표시한 이미지를 리턴하는 메소드     
'''

import plt
import numpy as np
from skimage.transform import resize

def black_white_windowing(dicom_pixel_array):
    threshold = 80

    lut_min = np.linspace(255, 255, num=15, endpoint=True)
    lut_mid = np.linspace(0, 0, num=threshold-15, endpoint=True)
    lut_max = np.linspace(255, 255, num=256-threshold, endpoint=True)

    lut = np.concatenate((lut_min, lut_mid, lut_max))

    dicom_pixel_array = lut[dicom_pixel_array]

    pixel_array = np.stack((dicom_pixel_array,)*3, axis=-1)

    data = plt.make_image(img=pixel_array, title="Windowing to Gray Scale\n")

    return data



def coord_windowing(dicom_pixel_array):
    pixel_array = resize(dicom_pixel_array, (512, 512), preserve_range=True)

    pixel_mean = int(sum([sum(row) for row in pixel_array]) / (512 * 512))
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

    lut_min = np.linspace(255, 255, num=(pixel_mean-40), endpoint=True)
    lut_mid = np.linspace(0, 0, num=80, endpoint=True)
    lut_max = np.linspace(255, 255, num=256-(pixel_mean+20), endpoint=True)

    lut = np.concatenate((lut_min, lut_mid, lut_max))
    pixel_array = lut[pixel_array.astype(np.uint8)]
    pixel_array = [[pixel_array[i][j] for j in range(0, len(pixel_array[i]), 8)] for i in range(0, len(pixel_array), 8)]

    xs = [x for x in range(len(pixel_array))]
    y = [sum(row)/len(pixel_array[0]) for row in pixel_array]
    start_idx = -1
    end_idx = len(pixel_array)
    for i, x in enumerate(y):
        if i == 0 or i == len(pixel_array) - 1:
            continue
        if y[i] <= 180 and y[i-1] >= 180:
            start_idx = i
        if y[i] <= 220 and y[i+1] >= 220:
            end_idx = i

    start_idx = max(0, int(start_idx * len(dicom_pixel_array) / 64))
    end_idx = min(len(dicom_pixel_array), int(end_idx * len(dicom_pixel_array) / 64))
    pixel_array = np.stack((dicom_pixel_array,)*3, axis=-1)

    for j in range(0, len(pixel_array[0])):
        for k in range(0, 15):
            if start_idx+k-7 >= 0 and start_idx+k-7 < len(pixel_array):
                pixel_array[start_idx+k-7][j] = [255, 255, 0]
            if end_idx+k-7 >= 0 and end_idx+k-7 < len(pixel_array):
                pixel_array[end_idx+k-7][j] = [255, 255, 0]

    data = plt.make_image(img=pixel_array, title="Anatomical Imaging Range\n" ,ticks=[0, start_idx, end_idx, len(pixel_array) - 1])

    return data
