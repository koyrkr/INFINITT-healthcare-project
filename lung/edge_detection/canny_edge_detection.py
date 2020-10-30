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
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from PIL import Image, ImageEnhance
from tqdm import tqdm
import scipy
import cv2
import time
import pydicom


# In[2]:


plt.rcParams.update({'font.size': 25})
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.figsize'] = [20, 10]


# In[3]:


## 0. Gather Images

img_list = []  # create an empty list

input_path = '../dataset/siim/input_sample_png'
output_path = '../dataset/siim/output_sample_png/'

for dirName, subdirList, fileList in os.walk(input_path):
    for filename in fileList:
        img_list.append(os.path.join(dirName,filename))


# In[4]:


## 1. Read Images
        
n = len(img_list)
arr1 = np.zeros((n, 512, 512))
arr2 = np.zeros((n, 512, 512))

with tqdm(total=n) as t:
    for file_index in range(n):
        img = Image.open(img_list[file_index]).convert('L')     
#         contrast = ImageEnhance.Contrast(img)
#         img = contrast.enhance(2.0)
        arr1[file_index,:,:] = np.array(img.resize((512, 512), Image.LANCZOS))
        t.set_description("Reading Input Images: ")
        t.update(1)
    t.close()


# In[5]:


## 2. Display Images
def display_image(str_kernel):
    print('===== Dislaying %s Kernel Images =====' % str_kernel)
    print("===== Kernel: " + str_kernel + " =====\n")
    start = time.process_time()
    for i in range(5):
        print("Image %s" % i)
        plt.axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(arr1[i,:,:], cmap=plt.cm.bone)
        plt.subplot(1, 2, 2)
#         plt.imshow(arr2[i,:,:], cmap='Greys',  interpolation='nearest')
        plt.imshow(arr2[i,:,:], cmap=plt.cm.bone)
        plt.show()
        plt.close()
    print("===== Done Displaying %s Kernel Images =====\n\n" % str_kernel)
    end = time.process_time()
    print("ELAPSED TIME: %s" % (end - start))


# In[6]:


## 3. Save Images 
def save_image(str_kernel):
    with tqdm(total=len(img_list)) as t:
        for i in range(n):
            plt.axis('off')
            plt.imshow(arr2[i,:,:], cmap=plt.cm.bone)
            plt.savefig(output_path + str_kernel + str(i) + '.png', bbox_inches = 'tight')
            plt.close()
            t.set_description("Saving %s Images: " % str_kernel)
            t.update(1)
        t.close()


# In[7]:


## 4. Convolution
def convolution(kernel1, kernel2, str_kernel):
    with tqdm(total=n) as t:
        for i in range(n):
            arr2[i,:,:] = cv2.Canny((arr1[i,:,:]).astype(np.uint8),100,200)
            t.set_description("Processing %s Images: " % str_kernel)
            t.update(1)
        t.close()


# In[8]:


## 5. Test Canny
convolution(None, None, "Canny")
display_image("Canny")
save_image("canny")


# In[9]:


## 6. 2D array

def plot_canny(pixel_array):   
    # Make a random plot...
    fig = plt.figure(figsize=(12,12))
    fig.tight_layout()
    ax = fig.gca()
    
    start = time.process_time()
    res = cv2.resize(pixel_array, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    canny_array = cv2.Canny(res.astype(np.uint8), 100, 200)
    
    ax.imshow(canny_array, cmap=plt.cm.bone)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    end = time.process_time()
    t = end - start

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    ax.set_title("Canny Edge Detection\n")
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = resize(data, output_shape=(600, 600), preserve_range=True)
    data = data.astype(np.uint8)
    
    # np array and time elapsed
    return (data, t)


# In[11]:


path = '../dataset/siim/input_sample_dicom/'
file_path = path+os.listdir(path)[0]
pixel_array = pydicom.dcmread(file_path).pixel_array
plot_canny(pixel_array)


# In[ ]:





# In[ ]:




