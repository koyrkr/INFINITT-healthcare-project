'''
    def extract_metadata:
        주어진 DICOM 이미지 파일 디렉토리 속의 모든 이미지로부터 metadata를 추출한 뒤 CSV 파일 형태로 `path_to_csv` 위치에 저장하는 메소드
'''


import pydicom
import os, glob
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import csv
from tqdm import tqdm


path_to_csv = '../dataset/siim_dicom_sample_metadata.csv'

dicom_image_description = pd.read_csv("../dataset/dicom_image_description.csv")

def extract_metadata(path):
    '''
        path: DICOM image directory path
    '''
    with open(path_to_csv, 'w', newline ='') as csvfile:
        fieldnames = list(dicom_image_description["Description"])
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)
        
        dcm_files = []
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    dcm_files.append(os.path.join(dirName, filename))

        with tqdm(total=len(dcm_files)) as t:
            for n, img in enumerate(dcm_files):
                img = pydicom.read_file(img)
                rows = []
                for field in fieldnames:
                    if img.data_element(field) is None:
                        rows.append('')
                    else:
                        x = str(img.data_element(field)).replace("'", "")
                        y = x.find(":")
                        x = x[y+2:]
                        rows.append(x)
                writer.writerow(rows)

                t.set_description("Extracting Dicom Metadata: ")
                t.update(1)
            t.close()


extract_metadata('../dataset/siim/input_sample_dicom/')

