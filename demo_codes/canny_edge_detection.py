'''
    def plot_canny:
        원본 이미지에 Canny Edge Detection을 적용시킨 결과 이미지를 리턴하는 메소드
'''

import numpy as np
import plt
from skimage.transform import resize
import cv2


def plot_canny(pixel_array):    
    '''
         pixel_array: original input image
    '''
    h, w = pixel_array.shape

    res = cv2.resize(pixel_array, dsize=(int(512 * w / h), 512), interpolation=cv2.INTER_CUBIC)
    canny_array = cv2.Canny(res.astype(np.uint8), 100, 200)
    canny_array = np.stack((canny_array,)*3, axis=-1)

    data = plt.make_image(canny_array, "Canny Edge Detection\n")

    return data
