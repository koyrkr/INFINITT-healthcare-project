'''
    def windowing:
        pydicom의 apply_voi_lut를 이용하여 windowing한 이미지를 리턴하는 메소드
'''

from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import copy

# read windowing background image for fast converting
background_img = imread('windowing.png')
background_img = resize(background_img, (600, 600), preserve_range=True)
background_img = background_img.astype(np.uint8)

def windowing(img, dcmfile):
    background = copy.deepcopy(background_img)

    # apply windowing
    img = apply_voi_lut(img, dcmfile)
    pixel_array = img
    pixel_array = np.stack((pixel_array,)*3, axis=-1)
    h, w = img.shape
    size = (int(462 * h / w), 462) if h < w else (462, int(462 * w / h))

    pixel_array = resize(pixel_array, size, preserve_range=True)

    # update background with windowed image
    start = int((600 - len(pixel_array))/2)
    end = int((600 - len(pixel_array[0]))/2)
    background[start:start+len(pixel_array),end:end+len(pixel_array[0])] = pixel_array

    data = background.astype(np.uint8)

    return data
