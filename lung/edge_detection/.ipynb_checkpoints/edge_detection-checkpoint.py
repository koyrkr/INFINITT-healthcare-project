{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Reading Input Images: : 100%|██████████| 5/5 [00:01<00:00,  2.52it/s]\n",
      "Processing Identity Images: : 100%|██████████| 5/5 [00:00<00:00,  8.90it/s]\n",
      "Saving Identity Images: : 100%|██████████| 5/5 [00:01<00:00,  3.04it/s]\n",
      "Processing Horizontal Images: : 100%|██████████| 5/5 [00:00<00:00,  8.99it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "===== Dislaying Horizontal Kernel Images =====\n",
      "===== Kernel: Horizontal =====\n",
      "\n",
      "Image 0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ -6.   4.   2. ...  -4.  -6.   6.]\n",
      " [ -9.   6.   3. ...  -6.  -9.   9.]\n",
      " [ -9.   6.   3. ...  -6.  -9.   9.]\n",
      " ...\n",
      " [-13.   7.   7. ...  -3.  -6.   6.]\n",
      " [-14.   7.   8. ...  -3.  -6.   6.]\n",
      " [-10.   5.   6. ...  -2.  -4.   4.]]\n",
      "[[ -6.   4.   2. ...  -4.  -6.   6.]\n",
      " [ -9.   6.   3. ...  -6.  -9.   9.]\n",
      " [ -9.   6.   3. ...  -6.  -9.   9.]\n",
      " ...\n",
      " [-13.   7.   7. ...  -3.  -6.   6.]\n",
      " [-14.   7.   8. ...  -3.  -6.   6.]\n",
      " [-10.   5.   6. ...  -2.  -4.   4.]]\n",
      "Image 1\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ -48.   25.    4. ...  -66. -147.   87.]\n",
      " [ -52.   26.    4. ...  -61. -141.   91.]\n",
      " [  -6.    1.    0. ...   14.    9.    9.]\n",
      " ...\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]]\n",
      "[[ -48.   25.    4. ...  -66. -147.   87.]\n",
      " [ -52.   26.    4. ...  -61. -141.   91.]\n",
      " [  -6.    1.    0. ...   14.    9.    9.]\n",
      " ...\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]]\n",
      "Image 2\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-27. -26.  -2. ...   7.  52.  17.]\n",
      " [-30. -29.  -2. ...   9.  55.  17.]\n",
      " [ -5.  -5.   0. ...   4.   4.   0.]\n",
      " ...\n",
      " [  0.   0.   0. ...   0.   0.   0.]\n",
      " [  0.   0.   0. ...   0.   0.   0.]\n",
      " [  0.   0.   0. ...   0.   0.   0.]]\n",
      "[[-27. -26.  -2. ...   7.  52.  17.]\n",
      " [-30. -29.  -2. ...   9.  55.  17.]\n",
      " [ -5.  -5.   0. ...   4.   4.   0.]\n",
      " ...\n",
      " [  0.   0.   0. ...   0.   0.   0.]\n",
      " [  0.   0.   0. ...   0.   0.   0.]\n",
      " [  0.   0.   0. ...   0.   0.   0.]]\n",
      "Image 3\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " ...\n",
      " [-202.  -52.  -33. ...    3.    6.    6.]\n",
      " [-199.  -50.  -32. ...    3.    6.    6.]\n",
      " [-132.  -33.  -21. ...    2.    4.    4.]]\n",
      "[[   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " ...\n",
      " [-202.  -52.  -33. ...    3.    6.    6.]\n",
      " [-199.  -50.  -32. ...    3.    6.    6.]\n",
      " [-132.  -33.  -21. ...    2.    4.    4.]]\n",
      "Image 4\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  0%|          | 0/5 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[  -6.  -30.  -25. ...  -90. -167.  117.]\n",
      " [  -6.  -32.  -27. ...  -92. -165.  120.]\n",
      " [   0.   -3.   -3. ...   -2.  -14.    3.]\n",
      " ...\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]]\n",
      "[[  -6.  -30.  -25. ...  -90. -167.  117.]\n",
      " [  -6.  -32.  -27. ...  -92. -165.  120.]\n",
      " [   0.   -3.   -3. ...   -2.  -14.    3.]\n",
      " ...\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]\n",
      " [   0.    0.    0. ...    0.    0.    0.]]\n",
      "===== Done Displaying Horizontal Kernel Images =====\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving Horizontal Images: : 100%|██████████| 5/5 [00:01<00:00,  3.17it/s]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import os,sys\n",
    "from scipy.signal import convolve2d \n",
    "import matplotlib.pyplot as plt\n",
    "from skimage.segmentation import slic\n",
    "from skimage.segmentation import mark_boundaries\n",
    "from PIL import Image\n",
    "from tqdm import tqdm\n",
    "from time import sleep\n",
    "\n",
    "\n",
    "## 0. Gather Images\n",
    "\n",
    "# print('===== READING SAMPLE DATA =====')\n",
    "# !wget -O 'sample.png' https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Felix_Titling_sample.png/1280px-Felix_Titling_sample.png\n",
    "# img_list = ['sample.png'] # test \n",
    "# print('===== DONE READING SAMPLE DATA =====\\n\\n')\n",
    "\n",
    "\n",
    "path = \"./img/\"\n",
    "img_list = []  # create an empty list\n",
    "\n",
    "for dirName, subdirList, fileList in os.walk(path):\n",
    "    for filename in fileList:\n",
    "        img_list.append(os.path.join(dirName,filename))\n",
    "\n",
    "        \n",
    "## 1. Read Images\n",
    "        \n",
    "# n = len(img_list)\n",
    "n = 5\n",
    "arr1 = np.zeros((n, 512, 512))\n",
    "arr2 = np.zeros((n, 512, 512))\n",
    "\n",
    "with tqdm(total=n) as t:\n",
    "    for file_index in range(n):\n",
    "        sleep(0.1)\n",
    "        img = Image.open(img_list[file_index])\n",
    "        arr1[file_index,:,:] = np.array(img.resize((512, 512), Image.LANCZOS))\n",
    "        t.set_description(\"Reading Input Images: \")\n",
    "        t.update(1)\n",
    "    t.close()\n",
    "\n",
    "\n",
    "## 2. Display Images\n",
    "def display_image(str_kernel):\n",
    "    print('===== Dislaying %s Kernel Images =====' % str_kernel)\n",
    "    print(\"===== Kernel: \" + str_kernel + \" =====\\n\")\n",
    "    for i in range(n):\n",
    "        print(\"Image %s\" % i)\n",
    "        plt.axis('off')\n",
    "        plt.imshow(arr2[i,:,:], cmap = 'gray')\n",
    "        plt.show()\n",
    "        print(convolve2d(arr1[i,:,:], horizontal, mode='same'))\n",
    "        print(arr2[i,:,:])\n",
    "    print(\"===== Done Displaying %s Kernel Images =====\\n\\n\" % str_kernel)     \n",
    "    \n",
    "            \n",
    "## 3. Save Images \n",
    "def save_image(str_kernel):\n",
    "    with tqdm(total=n) as t:\n",
    "        for i in range(n):\n",
    "            sleep(0.1)\n",
    "            plt.axis('off')\n",
    "            plt.imshow(arr2[i,:,:], cmap = 'gray')\n",
    "            plt.savefig('edge_detection/' + str_kernel + str(i) + '.png', bbox_inches = 'tight')\n",
    "            t.set_description(\"Saving %s Images: \" % str_kernel)\n",
    "            t.update(1)\n",
    "        t.close()\n",
    "    \n",
    "    \n",
    "## 4. Convolution\n",
    "def convolution(kernel1, kernel2, str_kernel):\n",
    "    with tqdm(total=n) as t:\n",
    "        for i in range(n):\n",
    "            sleep(0.1)\n",
    "            temp1 = convolve2d(arr1[i,:,:],kernel1,mode='same')\n",
    "            if kernel2 is not None:\n",
    "                temp2 = convolve2d(arr1[i,:,:],kernerl2,mode='same')\n",
    "\n",
    "            if str_kernel in (\"Identity\", \"Horizontal\", \"Vertical\"):\n",
    "                arr2[i,:,:] = temp1\n",
    "            elif str_kernel == \"Gaussian Blur\":\n",
    "                arr2[i,:,:] = scipy.ndimage.filters.gaussian_filter(arr1[i,:,:], sigma = 10)\n",
    "            elif str_kernel == \"Super Pixel\":\n",
    "                segments = slic(arr1[i,:,:], n_segments = 50, sigma = 10)\n",
    "                arr2[i,:,:] = mark_boundaries(arr1[i,:,:], segments)\n",
    "            elif str_kernel in (\"Gradient Magnitude\", \"Sobel Gradient Magnitude\"):\n",
    "                arr2[i,:,:] = np.sqrt(temp1**2 + temp2**2)\n",
    "            elif str_kernel in (\"Gradient Direction\", \"Sobel Gradient Direction\"):\n",
    "                arr2[i,:,:] = np.arctan(temp1/temp2)\n",
    "            elif str_kernel == \"Sharpening\":\n",
    "                arr2[i,:,:] = np.sqrt(temp1**2 + temp2**2) + arr1[i,:,:]\n",
    "            t.set_description(\"Processing %s Images: \" % str_kernel)\n",
    "            t.update(1)\n",
    "        t.close()\n",
    "            \n",
    "        \n",
    "## 5. Kernels\n",
    "\n",
    "# 5-1. Identity\n",
    "identity = np.array([\n",
    "    [0,0,0],\n",
    "    [0,1,0],\n",
    "    [0,0,0]\n",
    "])\n",
    "\n",
    "convolution(identity, None, \"Identity\")\n",
    "save_image(\"Identity\")\n",
    "\n",
    "# 2. Edge Detection (Horizontal)\n",
    "horizontal = np.array([\n",
    "    [-1,0,1],\n",
    "    [-1,0,1],\n",
    "    [-1,0,1]\n",
    "])\n",
    "\n",
    "convolution(horizontal, None, \"Horizontal\")\n",
    "display_image(\"Horizontal\")\n",
    "save_image(\"Horizontal\")\n",
    "\n",
    "# 3. Edge Detection (Vertical)\n",
    "vertical = np.array([\n",
    "    [-1,-1,-1],\n",
    "    [0,0,0],\n",
    "    [1,1,1]\n",
    "])\n",
    "\n",
    "# display_image(vertical, \"Vertical\")\n",
    "\n",
    "# 4. Gradient Magnitude\n",
    "kernerl1 = np.array([\n",
    "    [-1,-1,-1],\n",
    "    [0,0,0],\n",
    "    [1,1,1]\n",
    "])\n",
    "\n",
    "kernerl2 = np.array([\n",
    "    [-1,0,1],\n",
    "    [-1,0,1],\n",
    "    [-1,0,1]\n",
    "])\n",
    "\n",
    "\n",
    "# 5. Gradient Direction\n",
    "kernerl1 = np.array([\n",
    "    [-1,-1,-1],\n",
    "    [0,0,0],\n",
    "    [1,1,1]\n",
    "])\n",
    "kernerl2 = np.array([\n",
    "    [-1,0,1],\n",
    "    [-1,0,1],\n",
    "    [-1,0,1]\n",
    "])\n",
    "\n",
    "\n",
    "# 6. Sobel Gradient Magnitude\n",
    "kernerl1 = np.array([\n",
    "    [1,2,1],\n",
    "    [0,0,0],\n",
    "    [-1,-2,-1]\n",
    "])\n",
    "kernerl2 = np.array([\n",
    "    [1,0,-1],\n",
    "    [2,0,-2],\n",
    "    [1,0,-1]\n",
    "])\n",
    "\n",
    "\n",
    "# 7. Sobel Gradient Direction\n",
    "kernerl1 = np.array([\n",
    "    [1,2,1],\n",
    "    [0,0,0],\n",
    "    [-1,-2,-1]\n",
    "])\n",
    "kernerl2 = np.array([\n",
    "    [1,0,-1],\n",
    "    [2,0,-2],\n",
    "    [1,0,-1]\n",
    "])\n",
    "\n",
    "\n",
    "# 8. Guassian Blur\n",
    "\n",
    "\n",
    "# 9. Sharpening\n",
    "kernerl1 = np.array([\n",
    "    [1,1,1],\n",
    "    [0,0,0],\n",
    "    [-1,-1,-1]\n",
    "])\n",
    "kernerl2 = np.array([\n",
    "    [1,0,-1],\n",
    "    [1,0,-1],\n",
    "    [1,0,-1]\n",
    "])\n",
    "\n",
    "\n",
    "# 11. Super Pixel\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
