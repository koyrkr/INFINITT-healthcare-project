# check_problematic_images.py
# Checking Out with Problematic Images
# Input : DICOM files (.dcm)
# Output : DICOM file name / Problem Cause

import pydicom
from pydicom import dcmread
import numpy as np
import png
import os, glob

from PIL import Image, ImageEnhance

dicom_folder_path = "../windowing/input_all_dicom/"

index = 0

def dicom_to_img(in_dataset, out_filename):
    shape = ds.pixel_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
    with open(out_filename, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)

# DICOM Input -> PNG

for filename in glob.glob(os.path.join(dicom_folder_path, '*.dcm')):
    print(filename)
    index += 1
    with open(filename, 'r'):
        ds = dcmread(filename)
        arr = ds.pixel_array
        new_input_filename = filename.replace("../windowing/input_all_dicom/", "")
        dicom_to_img(ds, '%s000%d_%s_cur.png' % ("../windowing/dicom_image_input/", index, new_input_filename))
        # print("BEFORE")
        # print(arr)

        threshold = 80

        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                # if arr[i][j] < 5: 
                #     arr[i][j] = 255  
                
                if arr[i][j] < threshold:
                    arr[i][j] = 0

                elif arr[i][j] >= threshold and arr[i][j] != 255:
                    arr[i][j] = 255

        # print("AFTER")
        # print(arr)

        white_cnt = 0
        black_cnt = 0
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                if arr[i][j] == 255:
                    white_cnt += 1

                else:
                    black_cnt += 1


        all_cnt = white_cnt + black_cnt
        print("white_cnt = ", str(white_cnt))
        print("black_cnt = ", str(black_cnt))
        print("all_cnt = ", str(all_cnt))

        new_output_filename = filename.replace("../windowing/input_all_dicom/", "")
        dicom_to_img(ds, '%s000%d_%s_mpl.png' % ("../windowing/dicom_image_output/", index, new_output_filename))

        problem_threshold = 0.8
        if float(white_cnt / all_cnt) > problem_threshold :
            print("!!!!!!Problematic Image : White Pixels Overflow!!!!!!", index)

        else:
            print("Reasonable Image")
