from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import glob, os

im1_list = sorted(glob.glob('ManualMask/leftMask/*.png'))
im2_list = sorted(glob.glob('ManualMask/rightMask/*.png'))
im3_list = sorted(glob.glob('CXR_png/*.png'))
im4_list = sorted(glob.glob('ManualMask/mask/*.png'))
print(len(im1_list))
print(len(im2_list))
print(len(im3_list))
print(len(im4_list))

for i in range(0, len(im4_list)):
    print(im4_list[i][16:])
    newaddr = 'mask/' + im4_list[i][16:]
    print(newaddr)
    y = imread(im4_list[i])
    x = resize(y, (4096, 4096), preserve_range=True)
    imsave(newaddr, x)
