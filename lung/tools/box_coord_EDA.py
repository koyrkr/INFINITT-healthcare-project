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
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from PIL import Image
import cv2
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as patches
from coordinates import Coordinate
import json


# In[2]:


plt.rcParams.update({'font.size': 25})


# In[3]:


coord_ans = pd.read_csv('../dataset/siim_box_coord.csv')


# In[4]:


pd.options.display.max_columns = None


# In[5]:


print(coord_ans.shape)
coord_ans.head()


# In[6]:


x1 = []
y1 = []
x2 = []
y2 = []

with tqdm(total=len(coord_ans)) as t:
    for i in range(len(coord_ans)):
        patientId = coord_ans['filename'][i][0:-4]
        dcm_file = '../dataset/siim/input_all_dicom/%s.dcm' % patientId
        dcm_data = pydicom.read_file(dcm_file)
        if coord_ans['region_count'][i] == 0:
            continue
        coord_json = coord_ans['region_shape_attributes'][i]
        coord = json.loads(coord_json)
        x1.append(coord.get('x'))
        y1.append(coord.get('y'))
        x2.append(coord.get('x') + coord.get('width'))
        y2.append(coord.get('y') + coord.get('height'))
        t.set_description("Reading Coordinates: ")
        t.update(1)
    t.close()


# In[7]:


print("min(x1): %s max(x1): %s\nmin(y1): %s max(y1): %s\nmin(x2): %s max(x2): %s\nmin(y2): %s max(y2): %s" % 
      (min(x1), max(x1), min(y1), max(y1), min(x2), max(x2), min(y2), max(y2)))


# In[8]:


# Bounding Box Sample

im = np.array(Image.open('../dataset/CHNCXR/input_all_png/CHNCXR_0555_1.png'), dtype=np.uint8)

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(im, cmap=plt.cm.bone)

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


# In[9]:


path = '../dataset/siim/input_sample_dicom/'
pixel_array = pydicom.dcmread(path+os.listdir(path)[0]).pixel_array
pixel_array = np.stack((pixel_array,)*3, axis=-1)


# In[14]:


def box_eda(pixel_array):
    # Make a random plot...
    fig = plt.figure(figsize=(12,12))
    fig.tight_layout()
    ax = fig.gca()
    ax.imshow(pixel_array, cmap=plt.cm.bone)
    ax.xaxis.set_visible(False)

    plt.axhline(y=min(y1)+1, color='r', linewidth=2)
    plt.axhline(y=max(y1)-5, color='r', linewidth=2)
    plt.axhline(y=min(y2), color='b', linewidth=2)
    plt.axhline(y=max(y2)-5, color='b', linewidth=2)

    plt.text(60, min(y1)+50, 'y1 min: {}'.format(min(y1)))
    plt.text(60, max(y1)-20, 'y1 max: {}'.format(max(y1)))
    plt.text(60, min(y2)+50, 'y2 min: {}'.format(min(y2)))
    plt.text(60, max(y2)-30, 'y2 max: {}'.format(max(y2)))

    ax.set_yticks([0, min(y1), max(y1), len(pixel_array) - 1])
    ax.set_title("Y-Coordinate Statistics\n")

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = resize(data, output_shape=(600, 600), preserve_range=True)
    data = data.astype(np.uint8)
    
    return (data, (min(y1), max(y1), min(y2), max(y2)))


# In[ ]:


box_eda(pixel_array)

