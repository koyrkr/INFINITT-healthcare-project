'''
    def make_image:
        image와 metadata를 받아 matplotlib.pyplot 형태로 출력하는 메소드
'''

import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize

plt.rcParams.update({'font.size': 25})

def make_image(img, title, **kwargs):
    '''
        img: original input image
        title: plot title
        img2: overlapping image
        ticks: list containing start index of image, start index of landmark, end index of landmark, end index of image
    '''

    # set plot features
    fig = plt.figure(figsize=(12,12))
    fig.tight_layout()
    ax = fig.gca()
    if 'img2' in kwargs:
        ax.imshow(img, alpha=0.8)
        ax.imshow(kwargs['img2'], alpha=0.5, interpolation='none')
    else:
        ax.imshow(img)
    if 'ticks' in kwargs:
        ax.xaxis.set_visible(False)
        ax.set_yticks(kwargs['ticks'])
    else:
        ax.axis('off')
    ax.set_title(title, fontsize=20 if len(title) > 30 else 25)


    # plot image to pixel array
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = resize(data, output_shape=(600, 600), preserve_range=True)
    data = data.astype(np.uint8)

    plt.clf()

    return data
