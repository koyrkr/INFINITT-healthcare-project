#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from shutil import copyfile
import pydicom
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


# In[2]:


metadata = pd.read_csv('../dataset/siim_dicom_metadata.csv')
given = pd.read_csv('../dataset/siim_train_rle.csv')


# In[3]:


metadata.rename(columns={'SOPInstanceUID':'image_id'}, inplace=True)


# In[4]:


print(metadata.shape)
metadata.head()


# In[5]:


print(given.shape)
given.head()


# In[6]:


merged = pd.merge(left=metadata, right=given)


# In[7]:


print(merged.shape)
merged.head()


# In[8]:


print(len(merged[merged['EncodedPixels'] == '-1']))
print(len(merged[merged['EncodedPixels'] != '-1']))


# In[19]:


for i, row in merged.iterrows():
    name = row['image_id'] + '.dcm'
    if row['EncodedPixels']== '-1':
        copyfile('../dataset/siim/input_all_dicom/'+name, '../dataset/siim/input_normal_dicom/'+name)
    else:
        copyfile('../dataset/siim/input_all_dicom/'+name, '../dataset/siim/input_pneumo_dicom/'+name)


# In[ ]:




