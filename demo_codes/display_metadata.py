'''
    def metadata:
        주어진 이미지에 대한 DICOM metadata를 제목으로 표기하여 이미지로 생성한 뒤 이를 표기된 metadata와 함께 리턴하는 메소드
'''

import pydicom
import numpy as np
import plt
from skimage.transform import resize

def metadata(file_path, pixel_array):
    '''
        file_path: input image file path
        pixel_array: original input image
    '''
    pixel_array = np.stack((pixel_array,)*3, axis=-1)

    img_data = pydicom.dcmread(file_path)

    info = 'ID: {}\nModality: {}    Age: {}    Sex: {}\nBody Part Examined: {}\nView Position: {}'.format(
            img_data.PatientID,
            img_data.Modality,
            img_data.PatientAge,
            img_data.PatientSex,
            img_data.BodyPartExamined,
            img_data.ViewPosition)

    data = plt.make_image(pixel_array, info)

    return (data, info)
