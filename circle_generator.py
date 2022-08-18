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
c_mat = np.ones((h, w,4), dtype=np.float32)

rr,cc = skimage.draw.disk((h//2,w//2),5)
c_mat[rr,cc,:4] =0
# c_mat[:,:,-1] = 1.0
plt.figure()
plt.imshow(c_mat)
np.save('cirv6b.npy',c_mat)
# f, axarr = plt.subplots(2,2)

h = 40
w = 40
seed = np.zeros([h, w, 16], np.float32)
seed[h//2, w//2, 3:] = 1.0
# plt.figure()
# plt.imshow(seed[:,:,5:8])
# x = np.zeros([n, size, size, CHANNEL_N], np.float32)
# x[:, size//2, size//2, 3:] = 1.0
TARGET_SIZE = 40
import requests
import io
import PIL.Image, PIL.ImageDraw
url = 'https://github.com/MishaRubanov/DNAHydrogelNeuralCellularAutomata/blob/main/notebooks/40x40watermelon.png?raw=true'

def load_image(url, max_size=TARGET_SIZE):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img

z = load_image(url)
plt.figure()
plt.imshow(z)

import tensorflow as tf
pad_target = c_mat[:,:,0]
def to_rgba(x):
    return x[...,:4]

def to_g(x):
    return x[...,0]

def loss_f2(x):
  return tf.reduce_mean(tf.square(to_g(x)-pad_target))#, [-2, -1])

c_mat2 = np.ones((h, w,4), dtype=np.float32)

rr,cc = skimage.draw.disk((h//2,w//2),3)
c_mat2[rr,cc,0] =0

z = loss_f2(seed)
loss = tf.reduce_mean(loss_f2(c_mat2))
l2 = np.array(loss)
y1 = to_g(seed)
y2 = to_rgba(seed)
np.load('cirv4.npy',allow_pickle=True)
