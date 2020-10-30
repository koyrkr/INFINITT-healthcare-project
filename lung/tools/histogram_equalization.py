#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
import os, glob
import cv2
import pydicom
from glob import glob
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm


# In[2]:


# Read Files

input_dir = '../dataset/siim/input_sample_dicom/'
input_png_dir = '../dataset/siim/input_sample_png/'
patients = os.listdir(input_dir)
patients.sort()


dcm_files = []
def load_scan2(path):
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                dcm_files.append(os.path.join(dirName, filename))
    return dcm_files

first_patient = load_scan2(input_dir)

# Preprocess Dicom Image

def equalize(path):    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    plt.axis('off')
    img = pydicom.dcmread(path).pixel_array
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    hist_equalized = cv2.equalizeHist(img)
    clahe_equalized = clahe.apply(img)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))

    # Histogram Equalization
    ax1.imshow(img, cmap=plt.cm.bone)
    ax1.set_title('Original')

    ax2.imshow(hist_equalized, cmap=plt.cm.bone)
    ax2.set_title('Histogram Equalization')

    ax3.imshow(clahe_equalized, cmap=plt.cm.bone)
    ax3.set_title('Clahe Histogram Equalization')

    addr = path.replace('.dcm', '.png')
    addr = addr.replace('/input_sample_dicom/', '/input_equalized_png/')
    plt.savefig(addr)
    plt.show()
    plt.close()


# In[6]:


equalize(dcm_files[3])

