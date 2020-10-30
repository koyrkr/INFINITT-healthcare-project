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


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            rounded = [round(num, 3) for num in dataset.PixelSpacing]
            print("Pixel spacing....:", rounded)
    print()


# In[3]:


def plot_pixel_array(dataset, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


# In[4]:


path = '../dataset/siim/input_sample_dicom/*.dcm'
img_list = glob.glob(path)
for file_path in glob.glob(path):
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    plot_pixel_array(dataset)


# In[7]:


# Sample
dicom_file_dataset = pydicom.read_file(img_list[1])
dicom_file_dataset


# In[ ]:




