from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import glob, os

im1_list = sorted(glob.glob('ManualMask/leftMask/*.png'))
im2_list = sorted(glob.glob('ManualMask/rightMask/*.png'))
im3_list = sorted(glob.glob('CXR_png/*.png'))
print(len(im1_list))
print(len(im2_list))

# y = imread(im3_list[0])
# plt.subplot(121)
# plt.imshow(y)
# x = resize(y, (4096, 4096), preserve_range=True)
# imsave('test.png', x)
# plt.subplot(122)
# plt.imshow(x)
# plt.show()

for i in range(0, len(im1_list)):
    im1 = Image.open(im1_list[i])
    im2 = Image.open(im2_list[i])

    pix1 = im1.load()
    pix2 = im2.load()
    print(im1.size)
    print(im2.size)
    print(im1_list[i][20:])
    newaddr = 'ManualMask/mask/' + im1_list[i][20:]
    print(newaddr)
    imnew = Image.new(mode='L', size=im1.size)
    pixnew = imnew.load()
    for i in range(0, imnew.size[0]):
        for j in range(0, imnew.size[1]):
            pixnew[i,j] = max(pix1[i,j], pix2[i,j])
    imnew.save(newaddr)
