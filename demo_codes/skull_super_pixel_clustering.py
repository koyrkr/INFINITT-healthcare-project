'''
    def clustering:
        Super Pixel과 k-means clustering 기법을 적용한 이미지를 리턴하는 메소드
    def clustering_coord:
        feature extraction(clustering)으로 구한 skull의 윗 좌표, 아랫 좌표와 업데이트된 이미지를 리턴하는 메소드
'''

import numpy as np
import cv2
import plt
from skimage.transform import resize
from skimage.segmentation import slic, mark_boundaries

def clustering(pixel_array):
    '''
        pixel_array: original input image
    '''
    clusters = slic(pixel_array, compactness=0.01, n_segments=10, sigma=20)

    img = mark_boundaries(pixel_array, clusters, color=(0,0,0), mode='outer')

    data = plt.make_image(img, "Super Pixel + Clustering\n")

    return data

def clustering_coord(pixel_array):
    '''
        pixel_array: original input image
    '''
    h, w = pixel_array.shape

    clusters = slic(pixel_array, compactness=0.01, n_segments=10, sigma=20)

    cnt = len(np.unique(clusters))
    min = [(0, 0) for _ in range(cnt)]
    max = [(0, 0) for _ in range(cnt)]
    score = [0 for _ in range(cnt)]

    for i in range(cnt):
        for row_idx, row in enumerate(clusters):
            if i in row:
                min[i] = (row_idx, i)
                break
        for row_idx, row in enumerate(reversed(clusters)):
            if i in row:
                max[i] = (len(clusters) - 1 - row_idx, i)
                break

    min = sorted(min)
    max = sorted(max, reverse=True)
    # print(min, max)

    for k, (i, j) in enumerate(zip(min, max)):
        score[i[1]] += k
        score[j[1]] += k
    # print(score)

    fst_cluster, score_max = -1, -1
    for i, x in enumerate(score):
        if x > score_max:
            score_max = x
            fst_cluster = i
    # print(fst_cluster)

    start_idx = 0
    end_idx = 0

    for i, row in enumerate(pixel_array):
        if sum(row) / len(row) > 30:
            start_idx = i
            break

    for i in max:
        if i[1] == fst_cluster:
            end_idx = i[0]
            break

    pixel_array = np.stack((pixel_array,)*3, axis=-1)

    for j in range(0, len(pixel_array[0])):
        for k in range(0, 5):
            if start_idx+k-2 >= 0 and start_idx+k-2 < len(pixel_array) and end_idx+k-2 >= 0 and end_idx+k-2 < len(pixel_array):
                pixel_array[start_idx+k-2][j] = [255, 128, 0]
                pixel_array[end_idx+k-2][j] = [255, 128, 0]

    data = plt.make_image(img=pixel_array, title="Anatomical Imaging Range\n", ticks=[0, start_idx, end_idx, len(pixel_array) - 1])

    return data
