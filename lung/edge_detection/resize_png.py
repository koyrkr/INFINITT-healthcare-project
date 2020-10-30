#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for skull image resizing


# In[ ]:


#!/usr/bin/python
from PIL import Image
import os, sys

input_path = '../dataset/skull/original/'
output_path = '../dataset/skull/resized/'
dirs = os.listdir(input_path)

def resize():
    for item in dirs:
        if os.path.isfile(input_path+item):
            im = Image.open(input_path+item)
            imResize = im.resize((512, 512), Image.ANTIALIAS)
            imResize.save(os.path.join(output_path, item), 'JPEG', quality=90)

resize()

