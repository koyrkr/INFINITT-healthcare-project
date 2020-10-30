#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom
import os, glob
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm


# In[2]:


def show_dicom_images(data):
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i, img_path in enumerate(glob.glob(path)):
        img_data = pydicom.read_file(img_path)
        data_row_img = pydicom.dcmread(img_path)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {}    Age: {}    Sex: {}\nBody Part Examined: {}\nView Position: {}'.format(
            img_data.PatientID, 
            img_data.Modality, 
            img_data.PatientAge, 
            img_data.PatientSex,
            img_data.BodyPartExamined,
            img_data.ViewPosition))
    plt.show()


# In[3]:


path = '../dataset/siim/input_sample_dicom/*.dcm'

show_dicom_images(path)


# In[ ]:


# Not Labeled

start = 1   # Starting index of images
num_img = 4 # Total number of images to show

fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
for q, file_path in enumerate(glob.glob(path)[start:start+num_img]):
    dataset = pydicom.dcmread(file_path)    
    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

