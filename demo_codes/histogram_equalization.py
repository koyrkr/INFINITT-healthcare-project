'''
    def hist_equalize:
        주어진 이미지에 대해 histogram equalization을 적용한 이미지를 리턴하는 메소드
    def clahe_hist_equalize:
        주어진 이미지에 대해 Clahe histogram equalization을 적용한 이미지를 리턴하는 메소드
'''

import numpy as np
import cv2
import pydicom
import plt
from skimage.transform import resize

def hist_equalize(pixel_array):
    '''
        pixel_array: original input image
    '''
    hist_equalized = cv2.equalizeHist(pixel_array)
    hist_equalized = np.stack((hist_equalized,)*3, axis=-1)

    data = plt.make_image(hist_equalized, "Histogram Equalized\n")

    return data


def clahe_hist_equalize(pixel_array):
    '''
        pixel_array: original input image
    '''
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_hist_equalized = clahe.apply(pixel_array)
    clahe_hist_equalized = np.stack((clahe_hist_equalized,)*3, axis=-1)

    data = plt.make_image(clahe_hist_equalized, "Clahe Histogram Equalized\n")

    return data
