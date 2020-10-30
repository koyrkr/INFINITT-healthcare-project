#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import os, glob
get_ipython().run_line_magic('matplotlib', 'inline')
import PIL
import scipy.ndimage
from tqdm import tqdm


dcm_dir = '../dataset/siim/input_sample_dicom/'
png_dir = '../dataset/siim/input_sample_png/'
dataset_name = 'siim'

patients = os.listdir(dcm_dir)
patients.sort()

dcm_files = []
def load_scan2(path):
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                dcm_files.append(os.path.join(dirName, filename))
    return dcm_files

first_patient = load_scan2(dcm_dir)


# In[1]:


# Array Construction Using Reference Image
ref = pydicom.read_file(dcm_files[0])
ConstPixelDims = (int(ref.Rows), int(ref.Columns), len(dcm_files))
ConstPixelSpacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]))

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])

ArrayDicom = np.zeros(ConstPixelDims, dtype=ref.pixel_array.dtype)


# In[3]:


# Read Dicom Images

with tqdm(total=len(dcm_files)) as t:
    for dcm_file in dcm_files:
        ds = pydicom.read_file(dcm_file)
        ArrayDicom[:,:,dcm_files.index(dcm_file)] = ds.pixel_array
        t.set_description("Reading Dicom Images: ")
        t.update(1)
    t.close()


# In[4]:


# Display Sample Image

plt.figure(dpi=1600)
plt.axes().set_aspect('equal', 'datalim')
plt.set_cmap(plt.gray())
plt.pcolormesh(x, y, np.flipud(ArrayDicom[:,:,1]))
plt.show()


# In[5]:


# Convert

with tqdm(total=len(dcm_files)) as t:
    for n, image in enumerate(dcm_files):

        # Convert Dicom to PNG
        ds = pydicom.dcmread(os.path.join(image))
        pixel_array_numpy = ds.pixel_array
        image = image.replace('.dcm', '.png')
        image = image.replace(dcm_dir, png_dir)
        cv2.imwrite(os.path.join(image), pixel_array_numpy)

        t.set_description("Converting Dicom Images: ")
        t.update(1)
    t.close()


# In[6]:


# Simple Display

image_path = dcm_files[0]
ds = pydicom.dcmread(image_path)
plt.imshow(ds.pixel_array)
plt.show()

