import os, glob, numpy

from PIL import Image
from numpy import asarray

input_folder_path = "../../dataset/siim/output_sample_png/"
output_folder_path = "../../dataset/siim/output_sample_png_be/"

for filename in glob.glob(os.path.join(input_folder_path, '*.png')):
    print(filename)
    with open(filename, 'r'):
        image = Image.open(filename)
        image_gray = image.convert('LA')
        arr = asarray(image_gray)
        arr_copy = arr.copy()

        for i in range(0, arr_copy.shape[0]):
            for j in range(0, arr_copy.shape[1]):
                if arr_copy[i][j][0] < 5:
                    arr_copy[i][j][0] = 255


        image_new = Image.fromarray(arr_copy)
        new_filename = filename.replace(input_folder_path, "")
        print(new_filename)
        image_new.save(output_folder_path + new_filename)

 
    