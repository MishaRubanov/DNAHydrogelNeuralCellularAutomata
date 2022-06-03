# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:22:02 2022

@author: rmish
"""

#%%Functions and initalization:
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import scipy.signal

gens = 7
n = 11
global X
global img      
global A

#%%3x3 matrix
A = np.zeros((n,n))
for i in [2,4,6]:
    for j in [2,4,6]:
        A[i,j] = 1
        
xgens = np.zeros((n,n,gens+1))
xgens[:,:,0] = A


''' define growth function with growth/shrink ranges '''
def growthAND(U):
  return 0+ (U>=2) - (U<2)

def growthOR(U):
    return 0 + (U>=1)-(U<1)

def growthstrictAND(U):
  return 0+ (U>=3) - (U<3)

def growthNAND(U):
    return 0 + (U<1) - (U>=1)
K = np.asarray([[1,1,1], [1,0,1], [1,1,1]])
K_sum = np.sum(K)


def update(A,i):
  ''' use convolution '''
  U = scipy.signal.convolve2d(A, K, mode='same', boundary='fill', fillvalue = 0)
  N = np.clip(A + growthstrictAND(U), 0, 1)  
  return N

fig = plt.figure(dpi=150)

for i in range(1,gens+1):    
    xgens[:,:,i] = update(xgens[:,:,i-1],i)
    ax = fig.add_subplot(3,5,i)
    sx,sy = np.nonzero(xgens[:,:,i-1])
    plt.scatter(sx,sy,10,cmap='binary')
    plt.xlim([0,n])
    plt.ylim([0,n])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


