'''
    def landmark:
        FCN 모델로 예측한 이미지와 원본 이미지를 겹친 새로운 이미지를 만들어 리턴하는 메소드

    def coord:
        FCN 모델로 예측한 이미지를 받아 BFS 알고리즘을 사용하여 두 클러스터를 뽑고 구한 두 좌표와 원본 이미지에 두 좌표를 표시한 이미지를 리턴하는 메소드
'''

from collections import deque
import numpy as np
import plt
from skimage.transform import resize
from skimage.io import imsave

def landmark(img, pixel_array):
    '''
        img: mask image
        pixel_array: original input image
    '''

    # make mask to red border line
    img = np.stack([img, img/512, img/512, img], axis=-1)
    pixel_array = np.stack((pixel_array,)*3, axis=-1)

    # make overlapped image
    data = plt.make_image(img=pixel_array, img2=img, title="Landmark Extraction\n", ticks=[0, len(img) - 1])

    return data

def coord(img, pixel_array):
    '''
        img: mask image
        pixel_array: original input image
    '''
    h, w = pixel_array.shape

    # make image 1/16 resolution
    dst = np.array([[img[i][j] for j in range(0, len(img[i]), int(w / 64))] for i in range(0, len(img), int(h / 64))])

    # initialize start index and end index of landmark
    start_idx = -1
    end_idx = len(dst)

    # BFS code for finding largest two clusters
    group_table = dict()
    map = [[0 if dst[i][j] <= 200 else -1 for j in range(0, len(dst[i]))] for i in range(0, len(dst))]
    group_num = 0
    total_num = -sum([sum(map[i]) for i in range(len(map))])
    checked_num = 0

    while checked_num < total_num:
        group_num += 1
        x, y = (-1, -1)
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == -1:
                    x, y = (i, j)
                    break
            if x != -1 and y != -1:
                break
        queue = deque()
        queue.append((x, y))
        while queue:
            item = queue.popleft()
            # print(item)
            if map[item[0]][item[1]] == -1:
                map[item[0]][item[1]] = group_num
                checked_num += 1
                group_table[group_num] = 0 if group_num not in group_table else group_table[group_num] + 1
                if item[0] > 0 and map[item[0]-1][item[1]] == -1:
                    queue.append((item[0]-1, item[1]))
                if item[0] < len(map) - 1 and map[item[0]+1][item[1]] == -1:
                    queue.append((item[0]+1, item[1]))
                if item[1] > 0 and map[item[0]][item[1]-1] == -1:
                    queue.append((item[0], item[1]-1))
                if item[1] < len(map[0]) - 1 and map[item[0]][item[1]+1] == -1:
                    queue.append((item[0], item[1]+1))
    sorted_group_table = sorted(group_table.items(), key=(lambda x: x[1]), reverse=True)
    it = iter(sorted_group_table)
    start_idx_1, end_idx_1 = (len(map), -1)
    start_idx_2, end_idx_2 = (len(map), -1)
    fst_group = next(it, None)
    if fst_group:
        for i, j in zip(range(len(map)), reversed(range(len(map)))):
            if start_idx_1 == len(map) and fst_group[0] in map[i]:
                start_idx_1 = i
            if end_idx_1 == -1 and fst_group[0] in map[j]:
                end_idx_1 = j
            if start_idx_1 != len(map) and end_idx_1 != -1:
                break
    snd_group = next(it, None)
    if snd_group:
        for i, j in zip(range(len(map)), reversed(range(len(map)))):
            if start_idx_2 == len(map) and snd_group[0] in map[i]:
                start_idx_2 = i
            if end_idx_2 == -1 and snd_group[0] in map[j]:
                end_idx_2 = j
            if start_idx_2 != len(map) and end_idx_2 != -1:
                break

    # multiply 16 to start index and end index as BFS runned in 1/16 resolution
    start_idx = min(start_idx_1, start_idx_2) * int(h / 64)
    end_idx = max(end_idx_1, end_idx_2) * int(h / 64)

    pixel_array = np.stack((pixel_array,)*3, axis=-1)

    # make line on input image
    for j in range(0, len(pixel_array[0])):
        for k in range(0, 15):
            if start_idx+k-7 >= 0 and start_idx+k-7 < len(pixel_array) and end_idx+k-7 >= 0 and end_idx+k-7 < len(pixel_array):
                pixel_array[start_idx+k-7][j] = [255, 0, 0]
                pixel_array[end_idx+k-7][j] = [255, 0, 0]

    data = plt.make_image(img=pixel_array, title="Anatomical Imaging Range\n", ticks=[0, start_idx, end_idx, len(pixel_array) - 1])

    # return output image and tuple of start index and end index
    return (data, (start_idx, end_idx))

def get_only_coord(img, h, w):
    '''
        img: 512 * 512 mask image
    '''

    # make image 64 * 64 resolution
    dst = np.array([[img[i][j] for j in range(0, len(img[i]), 8)] for i in range(0, len(img), 8)])

    # initialize start index and end index of landmark
    start_idx = -1
    end_idx = len(dst)

    # BFS code for finding largest two clusters
    group_table = dict()
    map = [[0 if dst[i][j] <= 200 else -1 for j in range(0, len(dst[i]))] for i in range(0, len(dst))]
    group_num = 0
    total_num = -sum([sum(map[i]) for i in range(len(map))])
    checked_num = 0

    while checked_num < total_num:
        group_num += 1
        x, y = (-1, -1)
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] == -1:
                    x, y = (i, j)
                    break
            if x != -1 and y != -1:
                break
        queue = deque()
        queue.append((x, y))
        while queue:
            item = queue.popleft()
            # print(item)
            if map[item[0]][item[1]] == -1:
                map[item[0]][item[1]] = group_num
                checked_num += 1
                group_table[group_num] = 0 if group_num not in group_table else group_table[group_num] + 1
                if item[0] > 0 and map[item[0]-1][item[1]] == -1:
                    queue.append((item[0]-1, item[1]))
                if item[0] < len(map) - 1 and map[item[0]+1][item[1]] == -1:
                    queue.append((item[0]+1, item[1]))
                if item[1] > 0 and map[item[0]][item[1]-1] == -1:
                    queue.append((item[0], item[1]-1))
                if item[1] < len(map[0]) - 1 and map[item[0]][item[1]+1] == -1:
                    queue.append((item[0], item[1]+1))
    sorted_group_table = sorted(group_table.items(), key=(lambda x: x[1]), reverse=True)
    it = iter(sorted_group_table)
    start_idx_1, end_idx_1 = (len(map), -1)
    start_idx_2, end_idx_2 = (len(map), -1)
    fst_group = next(it, None)
    if fst_group:
        for i, j in zip(range(len(map)), reversed(range(len(map)))):
            if start_idx_1 == len(map) and fst_group[0] in map[i]:
                start_idx_1 = i
            if end_idx_1 == -1 and fst_group[0] in map[j]:
                end_idx_1 = j
            if start_idx_1 != len(map) and end_idx_1 != -1:
                break
    snd_group = next(it, None)
    if snd_group:
        for i, j in zip(range(len(map)), reversed(range(len(map)))):
            if start_idx_2 == len(map) and snd_group[0] in map[i]:
                start_idx_2 = i
            if end_idx_2 == -1 and snd_group[0] in map[j]:
                end_idx_2 = j
            if start_idx_2 != len(map) and end_idx_2 != -1:
                break

    # multiply 16 to start index and end index as BFS runned in 64 * 64 resolution
    start_idx = min(start_idx_1, start_idx_2) * int(h / 64)
    end_idx = max(end_idx_1, end_idx_2) * int(w / 64)

    return (start_idx, end_idx)
