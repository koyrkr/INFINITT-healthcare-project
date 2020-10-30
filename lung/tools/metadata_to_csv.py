#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydicom
import os, glob
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import csv
from tqdm import tqdm


# In[2]:


# dcm_dir = '../dataset/siim/input_all_dicom/'
dcm_dir = '../dataset/siim/input_sample_dicom/'
png_dir = '../dataset/siim/input_all_png/'
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


# In[3]:


# Read Dicom Images

with tqdm(total=len(dcm_files)) as t:
    for dcm_file in dcm_files:
        ds = pydicom.read_file(dcm_file)
        t.set_description("Reading Dicom Images: ")
        t.update(1)
    t.close()


# In[8]:


# Extract

dicom_image_description = pd.read_csv("../dataset/dicom_image_description.csv")

# with open(os.path.join('../dataset/', dataset_name, '_dicom_metadata.csv'), 'w', newline ='') as csvfile:
with open('../dataset/' + dataset_name + '_dicom_metadata.csv', 'w', newline ='') as csvfile:
    fieldnames = list(dicom_image_description["Description"])
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(fieldnames)
    
    with tqdm(total=len(dcm_files)) as t:
        for n, image in enumerate(dcm_files):
            rows = []
            for field in fieldnames:
                if ds.data_element(field) is None:
                    rows.append('')
                else:
                    x = str(ds.data_element(field)).replace("'", "")
                    y = x.find(":")
                    x = x[y+2:]
                    rows.append(x)
            writer.writerow(rows)
            
            t.set_description("Extracting Dicom Metadata: ")
            t.update(1)
        t.close()


# In[ ]:




