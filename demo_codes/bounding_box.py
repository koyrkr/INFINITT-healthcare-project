'''
    def coord_stat:
        SIIM dataset의 bounding box data를 (bounding box data가 존재하는 경우에 한해) 읽어와서 저장하는 메소드

    def box_eda:
        SIIM dataset의 bounding box의 x, y 좌표값의 min, max 값을 구하여 리턴하고, Chest X-ray 예시 이미지 위에 y 좌표의 min, max 값을 표시한 이미지를 만들어 함께 리턴하는 메소드
        
    def bounding_box_all:
        CSV 파일 형태로 주어진 bounding box 좌표값 모두를 이미지 dataset 디렉토리에서 찾고, 각각의 이미지 위에 bounding box를 표시하여 이미지를 생성한 뒤 새로운 디렉터리에 저장하는 메소드
    
    def bounding_box_selected:
        주어진 (이미지 파일) 경로에 대해 bounding box data가 존재하는지 확인하고, 존재할 경우 원본 이미지 위에 bounding box를 표시하여 이미지를 생성한 뒤 (x1, y1, x2, y2) 좌표값과 함께 리턴하는 메소드. Bounding box data가 존재하지 않을 경우 None를 리턴한다.
'''

get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom
import os, glob
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches
import re
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


plt.rcParams.update({'font.size': 25})
pd.options.display.max_columns = None


coord_ans = pd.read_csv('../dataset/siim_box_coord.csv')
labels = pd.read_csv('../dataset/stage_2_train_labels.csv') # for RSNA
metadata = pd.read_csv('../dataset/stage_2_detailed_class_info.csv') # for RSNA


path = '../dataset/siim/input_all_dicom/'
file_path = path+os.listdir(path)[0]
pixel_array = pydicom.dcmread(file_path).pixel_array


x1 = []
y1 = []
x2 = []
y2 = []


def coord_stat(path):
    '''
        path: image directory path
    '''
    with tqdm(total=len(coord_ans)) as t:
        for i in range(len(coord_ans)):
            patientId = coord_ans['filename'][i][0:-4]
            dcm_file = path+'%s.dcm' % patientId
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


def box_eda(path, pixel_array):
    '''
        path: image directory path
        pixel_array: original input image
    '''
    # Make a random plot...
    fig = plt.figure(figsize=(12,12))
    fig.tight_layout()
    ax = fig.gca()
    ax.imshow(pixel_array, cmap=plt.cm.bone)
    ax.xaxis.set_visible(False)
    
    coord_stat(path)

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

    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = resize(data, output_shape=(600, 600), preserve_range=True)
    data = data.astype(np.uint8)
    
    return (data, (min(y1), max(y1), min(y2), max(y2)))


def bounding_box_all(path):
    '''
        path: image directory path
    '''
    with tqdm(total=len(coord_ans)) as t:
        for i, row in coord_ans.iterrows():
            name = row['filename'][:-4]
            file_path = path+name+'.dcm'
            if os.path.isfile(file_path):
                dataset = pydicom.dcmread(file_path)
                box_coord_dic = row['region_shape_attributes']
                coord = json.loads(box_coord_dic)

                fig, ax = plt.subplots(1)
                ax.imshow(dataset.pixel_array, cmap=plt.cm.bone)

                rect = patches.Rectangle((coord.get("x"), coord.get("y")), 
                                          coord.get("width"), coord.get("height"), linewidth=2, edgecolor='r', facecolor='none')

                ax.add_patch(rect)
                addr = file_path.replace('/input_all_dicom/', '/output_all_box/')
                addr = addr.replace('.dcm', '.png')
                plt.savefig(addr)
                plt.close()
                t.set_description("Saving Bounding Box Images: ")
            t.update(1)
        t.close()


def bounding_box_selected(file_path, pixel_array):
    '''
        path: single image file path
        pixel_array: original input image
    '''
    res = re.search(path+'(.*).dcm', file_path)
    if coord_ans.loc[coord_ans['filename'] == (res.group(1) + '.jpg')] is None:
        return None
    box_coord_dic = coord_ans.loc[coord_ans['filename'] == (res.group(1) + '.jpg'), 'region_shape_attributes'].item()
    coord = json.loads(box_coord_dic)

    fig = plt.figure(figsize=(12,12))
    fig.tight_layout()
    ax = fig.gca()
    ax.imshow(pixel_array, cmap=plt.cm.bone)

    rect = patches.Rectangle((coord.get("x"), coord.get("y")), 
                              coord.get("width"), coord.get("height"), linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    ax.set_xticks([0, coord.get("x"), coord.get("x")+coord.get("width"), len(pixel_array)-1])
    ax.set_yticks([0, coord.get("y"), coord.get("y")+coord.get("height"), len(pixel_array) - 1])
    ax.set_title("Bounding Box\n")

    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = resize(data, output_shape=(600, 600), preserve_range=True)
    data = data.astype(np.uint8)
    
    # (data, (x1, y1, x2, y2))
    return (data, (coord.get("x"), coord.get("y"), coord.get("x")+coord.get("width"), coord.get("y")+coord.get("height")))


# Usage
bounding_box_all(path)
bounding_box_selected(file_path, pixel_array)
box_eda(path, pixel_array)

