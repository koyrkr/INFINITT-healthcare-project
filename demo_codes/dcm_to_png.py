'''
    `dcm_dir`에 주어진 DICOM 이미지 파일 디렉토리 속의 DICOM 영상 각각을 png 형태로 변환하여 `png_dir`에 동일한 이름으로 저장하는 프로그램. png_dir에 해당하는 디렉토리는 프로그램 실행 이전에 만들어야 한다.
'''

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


ref = pydicom.read_file(dcm_files[0])
ConstPixelDims = (int(ref.Rows), int(ref.Columns), len(dcm_files))
ConstPixelSpacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]))

x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])

ArrayDicom = np.zeros(ConstPixelDims, dtype=ref.pixel_array.dtype)


with tqdm(total=len(dcm_files)) as t:
    for dcm_file in dcm_files:
        ds = pydicom.read_file(dcm_file)
        ArrayDicom[:,:,dcm_files.index(dcm_file)] = ds.pixel_array
        t.set_description("Reading Dicom Images: ")
        t.update(1)
    t.close()


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

