'''
    head dicom file을 만들어주는 코드
'''

import os
import tempfile
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from skimage.io import imsave, imread
from PIL import Image

IMG_NUM = 11

for i in range(IMG_NUM):
    # read any dicom file
    dcmfile = pydicom.dcmread("head_dicom/1.2.276.0.7230010.3.1.4.8323329.304.1517875162.301989.dcm")

    # read image
    pixel_data = Image.open("head_dicom/skull"+str(i)+".jpg").convert('L')
    pixel_data = np.array(pixel_data)
    dcmfile.decompress()

    # update dicom file with head image
    dcmfile.PixelData = pixel_data.tostring()
    dcmfile.Rows, dcmfile.Columns = pixel_data.shape
    dcmfile.BodyPartExamined = "HEAD"
    dcmfile.ViewPosition = "RL"
    dcmfile.SeriesDescription = "view: RL"
    dcmfile.PatientName = "head_" + str(i)
    dcmfile.PatientID = "head_" + str(i)
    # del dcmfile.PatientAge

    # save dicom file
    dcmfile.save_as('head_dicom/head_'+str(i)+'.dcm')

    print(dcmfile)
