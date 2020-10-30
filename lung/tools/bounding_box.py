#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom
import os, glob, os.path
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
import json


coord_ans = pd.read_csv('../dataset/siim_box_coord.csv')
labels = pd.read_csv('../dataset/stage_2_train_labels.csv') # for RSNA
metadata = pd.read_csv('../dataset/stage_2_detailed_class_info.csv') # for RSNA


# In[2]:


sample_only = True

if sample_only:
    path = '../dataset/siim/input_sample_dicom/'
else:
    path = '../dataset/siim/input_all_dicom/'


# In[3]:


coord_ans.head()


# In[4]:


labels.head()


# In[5]:


metadata.head()


# In[6]:


# Bounding Box Sample

im = np.array(Image.open('../dataset/CHNCXR/input_all_png/CHNCXR_0555_1.png'), dtype=np.uint8)

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(im, cmap='gray')

# Create a Rectangle patch
rect1 = patches.Rectangle((300, 350), 1830, 1790, linewidth=2, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((270, 360), 1680, 2000, linewidth=2, edgecolor='b', facecolor='none')
rect3 = patches.Rectangle((250, 380), 1700, 1800, linewidth=2, edgecolor='g', facecolor='none')
rect4 = patches.Rectangle((330, 360), 1500, 1500, linewidth=2, edgecolor='y', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)

plt.show()


# In[7]:


# search using CSV data in image directory

with tqdm(total=len(coord_ans)) as t:
    for i, row in coord_ans.iterrows():
        name = row['filename'][:-4]
        file_path = path+name+'.dcm'
        res = re.search(path+'(.*).dcm', file_path)
        if os.path.isfile(file_path):
            dataset = pydicom.dcmread(file_path)
            box_coord_dic = row['region_shape_attributes']
            coord = json.loads(box_coord_dic)

            fig, ax = plt.subplots(1)
            ax.imshow(dataset.pixel_array, cmap=plt.cm.bone)

            rect = patches.Rectangle((coord.get("x"), coord.get("y")), 
                                      coord.get("width"), coord.get("height"), linewidth=2, edgecolor='r', facecolor='none')

            ax.add_patch(rect)
            if sample_only:
                addr = file_path.replace('/input_sample_dicom/', '/output_sample_box/')
            else:
                addr = file_path.replace('/input_all_dicom/', '/output_all_box/')
            addr = addr.replace('.dcm', '.png')
            plt.savefig(addr)
            plt.close()
            t.set_description("Saving Bounding Box Images: ")
        t.update(1)
    t.close()


# In[14]:



def bounding_box_selected(file_path):
    res = re.search('../dataset/siim/input_all_dicom/(.*).dcm', file_path)
    
    if coord_ans.loc[coord_ans['filename'] == (res.group(1) + '.jpg')] is None:
        return None
    box_coord_dic = coord_ans.loc[coord_ans['filename'] == (res.group(1) + '.jpg'), 'region_shape_attributes'].item()
    coord = json.loads(box_coord_dic)
    fig, ax = plt.subplots(1)
    ax.imshow(pydicom.dcmread(file_path).pixel_array, cmap=plt.cm.bone)
    
    rect = patches.Rectangle((coord.get("x"), coord.get("y")), 
                              coord.get("width"), coord.get("height"), linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    addr = file_path.replace('/input_sample_dicom/', '/output_sample_box/')
    addr = addr.replace('.dcm', '.png')
    plt.savefig(addr)
    plt.show()
    plt.close()


# In[15]:


path = '../dataset/siim/input_all_dicom/'
file_path = path+os.listdir(path)[0]
bounding_box_selected(file_path)


# In[ ]:




