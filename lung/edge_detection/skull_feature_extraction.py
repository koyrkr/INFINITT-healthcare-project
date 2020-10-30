#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os,sys
from scipy.signal import convolve2d 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance
from tqdm import tqdm
import scipy
import cv2
import time
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.figsize'] = [20, 10]


## 0. Gather Images

img_list = []  # create an empty list

# print('===== READING SAMPLE DATA =====')
# !wget -O 'sample.png' https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Felix_Titling_sample.png/1280px-Felix_Titling_sample.png
# img_list = ['sample.png'] # test 
# print('===== DONE READING SAMPLE DATA =====\n\n')


input_path = '../dataset/skull/resized/'
output_path = '../dataset/skull/feature_extraction/'

for dirName, subdirList, fileList in os.walk(input_path):
    for filename in fileList:
        img_list.append(os.path.join(dirName,filename))
        
        
## 1. Read Images
        
n = len(img_list)
arr1 = np.zeros((n, 512, 512))
arr2 = np.zeros((n, 512, 512))

with tqdm(total=n) as t:
    for file_index in range(n):
        img = Image.open(img_list[file_index]).convert('L')     
#         contrast = ImageEnhance.Contrast(img)
#         img = contrast.enhance(2.0)
#         cdf = scipy.ndimage.histogram(img, min=0, max=256, bins=256).cumsum()
#         arr1[file_index,:,:] = cdf[np.array(img.resize((512, 512), Image.LANCZOS))] * 255
        arr1[file_index,:,:] = np.array(img.resize((512, 512), Image.LANCZOS))
        t.set_description("Reading Input Images: ")
        t.update(1)
    t.close()


## 2. Display Images
def display_image(str_kernel):
    print('===== Dislaying %s Kernel Images =====' % str_kernel)
    print("===== Kernel: " + str_kernel + " =====\n")
    start = time.process_time()
    for i in range(1):
        print("Image %s" % i)
        plt.axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
        plt.imshow(arr2[i,:,:], cmap='gray',  interpolation='nearest')
        plt.show()
        plt.close()
    print("===== Done Displaying %s Kernel Images =====\n\n" % str_kernel)
    end = time.process_time()
    print("ELAPSED TIME: %s" % (end - start))
    
            
## 3. Save Images 

save_adjacent = False

def save_image(str_kernel):
    with tqdm(total=n) as t:
        for i in range(n):
            plt.axis('off')
            if save_adjacent:
                plt.subplot(1, 2, 1)
                plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
                plt.subplot(1, 2, 2)
                plt.imshow(arr2[i,:,:], cmap='gray',  interpolation='nearest')
                plt.savefig(output_path + 'adjacent/' + str_kernel + str(i) + '.png', bbox_inches = 'tight')
            else:
                plt.imshow(arr2[i,:,:], cmap=plt.cm.bone)
                plt.savefig(output_path + str_kernel + str(i) + '.png', bbox_inches = 'tight')
            plt.close()
            t.set_description("Saving %s Images: " % str_kernel)
            t.update(1)
        t.close()
    
    
## 4. Convolution
def convolution(kernel1, kernel2, str_kernel):
    with tqdm(total=n) as t:
        for i in range(n):
            if kernel1 is not None:
                temp1 = convolve2d(arr1[i,:,:],kernel1,mode='same')
            if kernel2 is not None:
                temp2 = convolve2d(arr1[i,:,:],kernel2,mode='same')
                
            if str_kernel in ("Identity", "Horizontal", "Vertical"):
                arr2[i,:,:] = temp1
            elif str_kernel == "Gaussian Blur":
                arr2[i,:,:] = scipy.ndimage.filters.gaussian_filter(arr1[i,:,:], sigma = 10)
            elif str_kernel in ("Gradient Magnitude", "Sobel Gradient Magnitude"):
                arr2[i,:,:] = np.sqrt(temp1**2 + temp2**2)
            elif str_kernel in ("Gradient Direction", "Sobel Gradient Direction"):
                arr2[i,:,:] = np.arctan(temp1/temp2)
            elif str_kernel == "Sharpening":
                arr2[i,:,:] = np.sqrt(temp1**2 + temp2**2) + arr1[i,:,:]
            elif str_kernel == "Canny":
                arr2[i,:,:] = cv2.Canny((arr1[i,:,:]).astype(np.uint8),100,200)
            t.set_description("Processing %s Images: " % str_kernel)
            t.update(1)
        t.close()
        
####################
####################

## 5. Kernels

# 5-1. Identity
identity = np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])

convolution(identity, None, "Identity")
display_image("Identity")
save_image("identity")


# 5-2. Edge Detection (Horizontal)
horizontal = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

convolution(horizontal, None, "Horizontal")
display_image("Horizontal")
save_image("horizontal")


# 5-3. Edge Detection (Vertical)
vertical = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])

convolution(vertical, None, "Vertical")
display_image("Vertical")
save_image("vertical")


# 5-4. Gradient Magnitude
kernel1 = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])

kernel2 = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

convolution(kernel1, kernel2, "Gradient Magnitude")
display_image("Gradient Magnitude")
save_image("gradient_magnitude")


# 5-5. Sobel Gradient Magnitude
kernel1 = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])
kernel2 = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

convolution(kernel1, kernel2, "Sobel Gradient Magnitude")
display_image("Sobel Gradient Magnitude")
save_image("sobel_gradient_magnitude")


# 5-6. Gradient Direction
kernel1 = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])
kernel2 = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

convolution(kernel1, kernel2, "Gradient Direction")
display_image("Gradient Direction")
save_image("gradient_direction")


# 5-7. Sobel Gradient Direction
kernel1 = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])
kernel2 = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

convolution(kernel1, kernel2, "Sobel Gradient Direction")
display_image("Sobel Gradient Direction")
save_image("sobel_gradient_direction")


# 5-8. Guassian Blur
convolution(None, None, "Gaussian Blur")
display_image("Gaussian Blur")
save_image("gaussian_blur")


# 5-9. Sharpening
kernel1 = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])
kernel2 = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

convolution(kernel1, kernel2, "Sharpening")
display_image("Sharpening")
save_image("sharpening")


# 5-10. Super Pixel
for i in range(1):
    arr2[i,:,:] = slic(arr1[i,:,:], n_segments = 10, sigma = 5)
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(arr1[i,:,:], arr2[i,:,:], color=(0,0,0), mode='think'))
    plt.show()
    plt.close()
    
for i in range(n):
    arr2[i,:,:] = slic(arr1[i,:,:], n_segments = 10, sigma = 5)
    plt.axis('off')
    if save_adjacent:
        plt.subplot(1, 2, 1)
        plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(arr1[i,:,:], arr2[i,:,:], color=(0,0,0), mode='think'))
        plt.savefig(output_path + 'adjacent/super_pixel' + str(i) + '.png', bbox_inches = 'tight')
    else:
        plt.imshow(mark_boundaries(arr1[i,:,:], arr2[i,:,:], color=(0,0,0), mode='think'))
        plt.savefig(output_path + 'super_pixel' + str(i) + '.png', bbox_inches = 'tight')
    plt.close()

    
# 5-11. Canny
convolution(None, None, "Canny")
display_image("Canny")
save_image("canny")


# In[2]:


# 5-10. Super Pixel
for i in range(n):
    arr2[i,:,:] = slic(arr1[i,:,:], n_segments = 10, sigma = 15)
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(arr1[i,:,:], arr2[i,:,:], color=(0,0,0), mode='think'))
    plt.show()
    plt.close()
    
for i in range(n):
    arr2[i,:,:] = slic(arr1[i,:,:], n_segments = 10, sigma = 10)
    plt.axis('off')
    if save_adjacent:
        plt.subplot(1, 2, 1)
        plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(arr1[i,:,:], arr2[i,:,:], color=(0,0,0), mode='think'))
        plt.savefig(output_path + 'adjacent/super_pixel' + str(i) + '.png', bbox_inches = 'tight')
    else:
        plt.imshow(mark_boundaries(arr1[i,:,:], arr2[i,:,:], color=(0,0,0), mode='think'))
        plt.savefig(output_path + 'super_pixel' + str(i) + '.png', bbox_inches = 'tight')
    plt.close()


# In[ ]:




