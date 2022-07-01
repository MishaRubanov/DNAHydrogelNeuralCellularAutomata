# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:55:20 2022

@author: rmish
"""

import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

z = plt.imread('water-melon_v2.png')
image_resized = resize(z, (40,40),
                       anti_aliasing=True)

plt.imshow(image_resized)

plt.imsave('40x40watermelon.png',image_resized)