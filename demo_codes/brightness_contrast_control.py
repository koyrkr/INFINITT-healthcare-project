'''
    def bc_control:
        들어온 이미지에 대해서 Brightness와 Contrast를 조정하여 새로운 이미지를 리턴하는 메소드
'''

import plt
import numpy as np

from skimage.transform import resize
from PIL import Image, ImageEnhance

def bc_control(pixel_array, br, ct):
    '''
        pixel_array: input image
        br: brightness value
        ct: contrast value
    '''
    img = Image.fromarray(pixel_array)
    enhancer_ct = ImageEnhance.Contrast(img)
    img = np.array(enhancer_ct.enhance(ct))
    img = Image.fromarray(img)
    enhancer_br = ImageEnhance.Brightness(img)
    img = np.array(enhancer_br.enhance(br))
    img = np.stack((img,)*3, axis=-1)

    data = plt.make_image(img=img, title="Brightness Contrast Control\n")

    return data
