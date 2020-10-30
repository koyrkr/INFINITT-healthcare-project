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
import matplotlib.patches as patches
import re
from PIL import Image
from coordinates import Coordinate
import json


# In[2]:


coord_ans = pd.read_csv('../dataset/siim_box_coord.csv')
metadata = pd.read_csv('../dataset/siim_dicom_metadata.csv')


# In[3]:


path = '../dataset/siim/input_all_dicom/*.dcm'


# In[4]:


pd.options.display.max_columns = None


# In[10]:


# all siim data
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.countplot(metadata['PatientAge'], ax=ax)
plt.title("[All Data] Patient Age")
plt.xticks(rotation=90)
plt.show()


# In[11]:


fig, (ax) = plt.subplots(nrows=1)
plt.hist(metadata['PatientAge'], bins=20, range=[0, 100])
plt.title("[All Data] Patient Age")
plt.show()


# In[12]:


# siim data with coord answer
fig, (ax) = plt.subplots(nrows=1,figsize=(16,6))
sns.countplot(boxed['PatientAge'], ax=ax)
plt.title("[Test Data] Patient Age")
plt.xticks(rotation=90)
plt.show()


# In[13]:


# all siim data
sns.countplot(metadata['PatientSex'])
plt.title("[All Data] Patient Sex")
plt.show()


# In[14]:


# siim data with coord ans
sns.countplot(boxed['PatientSex'])
plt.title("[Test Data] Patient Sex")
plt.show()

