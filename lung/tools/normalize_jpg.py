#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os, glob
get_ipython().run_line_magic('matplotlib', 'inline')
import PIL
import time
import pandas as pd
import csv
import scipy.ndimage
from tqdm import tqdm


# Read Files

# input_dir = './rsna/test/'
# input_png_dir = './rsna/test_png/'
# input_dir = '../dataset/RSNA/input_sample_dicom'
# input_png_dir = '../dataset/RSNA/input_sample_png'
input_dcm_dir = '../dataset/siim/input_all_dcm'
problematic_jpg_dir = '../dataset/siim/input_problematic_jpg'
problematic_normalized_dir = '../dataset/siim/output_problematic_png'
# patients = os.listdir(input_dir)
# patients.sort()


problematic_files = []
def load_scan1(path):
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".jpg" in filename.lower():
                problematic_files.append(os.path.join(filename))
    return problematic_files

dcm_files = []
def load_scan2(files):
    for filename in problematic_files:
        patientId = filename[0:-4]
        dcm_file = '../dataset/siim/input_all_dicom/%s.dcm' % patientId
        dcm_files.append(dcm_file)

load_scan2(load_scan1(problematic_jpg_dir))

# Preprocess Dicom Image

def get_PIL_image(dataset):
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            mode = "I;16"
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated "
                            "and %d SamplesPerPixel" % (bits, samples))
            
        size = (dataset.Columns, dataset.Rows)

        im = PIL.Image.frombuffer(mode, size, dataset.pixel_array,
                                  "raw", mode, 0, 1)

    else:
        ew = dataset['WindowWidth']
        ec = dataset['WindowCenter']
        ww = int(ew.value[0] if ew.VM > 1 else ew.value)
        wc = int(ec.value[0] if ec.VM > 1 else ec.value)
        image = get_LUT_value(dataset.pixel_array, ww, wc)
        im = PIL.Image.fromarray(image).convert('L')
    return np.array(im)

def _get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)
    
def _window_image(img, window_center, window_width, slope, intercept):
    img = (img * slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    return np.clip(img, img_min, img_max) 

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def _read(path, desired_size=(512, 512)):    
    dcm = pydicom.dcmread(path)
    img = get_PIL_image(dcm)
    img = _normalize(img)
    if img.shape != (512, 512):
        img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)
    return img

windowed = np.zeros((len(dcm_files), 512, 512))
with tqdm(total=len(dcm_files)) as t:
    for n, dcm_file in enumerate(dcm_files):
        plt.axis('off')
        plt.subplot(1, 2, 1)
        img = _read(dcm_file, (512, 512))      
        windowed[n,:,:] = img
        plt.imshow(windowed[n,:,:], cmap=plt.cm.bone)
        addr = dcm_file.replace('.dcm', '.png')
        addr = addr.replace('/input_all_dicom/', '/normalized_problematic_png/')
        plt.savefig(addr)
        plt.close()
#         t.set_description("Saving Normalized Images: ")    
        t.update(1)
    t.close()
    
for i in range(len(dcm_files)):
#     print("Image %s : " % (i+1))
    print("Image %s: " % dcm_files[i])
    start = time.process_time()
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow((pydicom.read_file(dcm_files[i])).pixel_array, cmap=plt.cm.bone)
    plt.subplot(1, 2, 2)
    plt.imshow(windowed[i,:,:], cmap=plt.cm.bone)
#     plt.show()
    addr = dcm_files[i].replace('.dcm', '.png')
    addr = addr.replace('/input_all_dicom/', '/output_problematic_png/')
    plt.savefig(addr)
    plt.close()
    end = time.process_time()
    print("ELAPSED TIME: %s" % (end - start))


# In[6]:


get_ipython().system('python3 -m pip install opencv-python')


# In[ ]:




