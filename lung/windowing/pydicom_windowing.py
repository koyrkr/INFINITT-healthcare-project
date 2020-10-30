import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_voi_lut

import numpy as np
import png
import os, glob

import time

dicom_folder_path = "input_all_dicom_tmp/"
input_folder_path = "dicom_image_input_tmp/"
output_folder_path = "dicom_image_output_tmp/"
# file_name = 'dicom_input/dicom_lung_test.dcm'

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

index = 0
t = time.process_time()
for filename in glob.glob(os.path.join(dicom_folder_path, '*.dcm')):
    index += 1
    print(filename)
    with open(filename, 'r'):
        ds = dcmread(filename)
        # print(ds)
        arr = ds.pixel_array
        new_input_filename = filename.replace("input_all_dicom_tmp/", "")
        dicom_to_img(ds, '%s%s_%d_cur.png' % (input_folder_path, new_input_filename, index))

        # print(arr)
        # print(arr.shape)

        # Manipulating pixel_data
        # arr[arr < 90] = 0

        # Eliminating the background ..?
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                if arr[i][j] < 5:
                    arr[i][j] = 255

        # Now, will not manipulate the value 255
        # arr[arr < 100] = 0
        arr[arr < 80] = 255

        # Getting the neck line
        neck_start = 0
        neck_end = 0
        flag = False
        for i in range(0, 1):
            for j in range(0, arr.shape[1]-1):
                if arr[i][j] > arr[i][j+1] and arr[i][j] == 255 and flag == False:
                    neck_start = j
                    flag = True

        flag = False            
        for i in range(0, 1):
            for j in range(0, arr.shape[1]-1):
                if arr[i][j] < arr[i][j+1] and arr[i][j+1] == 255 and flag == False:
                    neck_end = j
                    flag = True

        # print("neck_start =", neck_start)
        # print("neck_end =", neck_end)

        # for i in range(0, arr.shape[0]):
        #     arr[i][neck_start] = 3
        #     arr[i][neck_end] = 3

        # Getting the starting line
        #arr[0][neck_start] ~ arr[arr.shape[0]-1][neck_start]
        #arr[0][neck_end] ~ arr[arr.shape[0]-1][neck_end]
        flag = False
        start_1 = 0
        itv = 8
        for i in range(0, arr.shape[0]-itv-1):
            if (arr[i][neck_start] == 255) and (arr[i+itv][neck_start] == 255) and flag == False:
                flag = True
                start_1 = i
        
        start_2 = 0
        flag = False
        for i in range(0, arr.shape[0]-itv-1):
            if (arr[i][neck_end] == 255) and (arr[i+itv][neck_end] == 255) and flag == False:
                flag = True
                start_2 = i
        
        start = min(start_1, start_2)
        print("start=", start)


        # Getting the ending line
        end = 1000
        cnt = 0
        itv = 100
        flag = False
        for i in range(arr.shape[0]-1, -1, -1):
            for j in range(0, arr.shape[1]-itv-1):
                if (arr[i][j] == 255) and (arr[i][j+itv] != 255) and flag == False:
                    cnt += 1

                if (arr[i][j] != 255) and (arr[i][j+itv] == 255) and flag == False:
                    cnt += 1

                if cnt > 3:
                    end = j
                    flag = True

        # Doesn't work with the cnt! (pixel is so small)
        # print("cnt=", cnt)
        print("end=", end)


        # Visualizing the lines
        thk = 10
        for i in range(0, thk):
            arr[start + i] = 3
            arr[end + i] = 3

        # print(arr)
        # print(pixel_data)
        new_output_filename = filename.replace("input_all_dicom_tmp/", "")
        dicom_to_img(ds, '%s%s_%d_mpl.png' % (output_folder_path, new_output_filename, index))

elapsed_time = time.process_time() - t

print("TIME CHECK (s)")
print(elapsed_time)
