# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:45:26 2022

@author: rmish
"""
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import skimage

# rr,cc = skimage.draw.circle_perimeter(5,5,3,shape=[11,11])
# c_mat = np.zeros((11, 11), dtype=np.uint8)

# c_mat[rr,cc] = 1
# plt.figure()
# plt.imshow(c_mat)
h = 40
w = 40
c_mat = np.zeros((h, w,3), dtype=np.uint8)

rr,cc = skimage.draw.disk((h//2,w//2),5)
c_mat[rr,cc,:] =255
plt.figure()
plt.imshow(c_mat)
np.save('cirv3.npy',c_mat)
# f, axarr = plt.subplots(2,2)

h = 72
w = 72
seed = np.zeros([h, w, 16], np.float32)
seed[h//2, w//2, 3:] = 1.0
# x = np.zeros([n, size, size, CHANNEL_N], np.float32)
# x[:, size//2, size//2, 3:] = 1.0