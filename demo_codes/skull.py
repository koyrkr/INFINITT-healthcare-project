'''
    def coord_windowing:
        windowing으로 구한 skull의 윗 좌표, 아랫 좌표와 업데이트된 이미지를 리턴하는 메소드
'''

from skimage.io import imread, imsave
from PIL import ImageEnhance, Image
import plt

import numpy as np

def coord_windowing(pixel_array):
    '''
        pixel_array: original input image
    '''

    # control brightness and contrast to make image clear
    img = Image.fromarray(pixel_array)
    enhancer_ct = ImageEnhance.Contrast(img)
    img = enhancer_ct.enhance(5)
    enhancer_br = ImageEnhance.Brightness(img)
    img = np.array(enhancer_br.enhance(5))

    # find first index of white pixel for each row
    white_pixel_coord_list = [0 for _ in range(img.shape[0])]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 255 and img[i][j + 1] >= 255 and img[i][j + 5] >= 255:
                white_pixel_coord_list[i] = j
                break

    # find start index of while pixel appearing
    start = 0
    for i in range(len(white_pixel_coord_list)):
        if white_pixel_coord_list[i] != 0:
            start = i
            break

    for i in range(start):
        white_pixel_coord_list[i] = white_pixel_coord_list[start]

    # find index of minimum index of white pixel start
    min = 100000
    min_idx = 0
    end = 0
    for i in range(len(white_pixel_coord_list)):
        if white_pixel_coord_list[i] <= min and white_pixel_coord_list[i] != 0:
            min = white_pixel_coord_list[i]
            min_idx = i

    # find inflection point after minimum index of white pixel start
    for i in range(min_idx, len(white_pixel_coord_list)-50):
        flag = True
        for j in range(50):
            if white_pixel_coord_list[i] < white_pixel_coord_list[i+j]:
                flag = False
        if flag:
            end = i
            break

    # draw landmark coordinate line on original input image
    img = np.stack((pixel_array,)*3, axis=-1)
    for i in range(img.shape[1]):
        for j in range(5):
            img[start + j - 1][i] = [255, 255, 0]
            img[end + j - 1][i] = [255, 255, 0]

    data = plt.make_image(img=img, title="Anatomical Imaging Range\n" ,ticks=[0, start, end, len(img) - 1])

    return data
