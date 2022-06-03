# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 12:25:12 2022

@author: rmish

The goal is to create a 2-channel CA that uses WTA calculations for each of the 
generations
"""

import numpy as np
import scipy.signal
import matplotlib.pylab as plt
import copy
from sklearn.utils import shuffle
import matplotlib.animation as animation

#%%Functions:
def figure_world(A, cmap='viridis'):
  global img
  fig = plt.figure()
  img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)
  plt.title('world A')
  return fig

def figure_asset(K, growth, cmap='viridis', K_sum=1, bar_K=False):
  global R
  K_size = K.shape[0];  K_mid = K_size // 2
  fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,1,2]})
  ax[0].imshow(K, cmap=cmap, interpolation="nearest", vmin=0)
  ax[0].title.set_text('kernel K')
  if bar_K:
    ax[1].bar(range(K_size), K[K_mid,:], width=1)
  else:
    ax[1].plot(range(K_size), K[K_mid,:])
  ax[1].title.set_text('K cross-section')
  ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
  if K_sum <= 1:
    x = np.linspace(0, K_sum, 1000)
    ax[2].plot(x, growth(x))
  else:
    x = np.arange(K_sum + 1)
    ax[2].step(x, growth(x))
  ax[2].axhline(y=0, color='grey', linestyle='dotted')
  ax[2].title.set_text('growth G')
  return fig

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    z = np.zeros(np.shape(mask),dtype=int)
    z[mask] = 1
    return z

def create_square_mask(h, w, length=None, center = None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))    
    if length is None:
        length = (int(w/4))
    z = z = np.zeros([h,w],dtype=int)
    lhalf = int(length/2)
    z[center[0]-lhalf:center[0]+lhalf,center[1]-lhalf:center[1]+lhalf] = 1
    return z

''' define growth function with growth/shrink ranges '''
def growth(U):
   return 0 + (U>=3) - (U<3)
''' define convolution kernel '''
K = np.asarray([[1,1,1], [1,0,1], [1,1,1]])
K_sum = np.sum(K)
def update(A,K):
  # global A
  ''' use convolution '''
  #U = sum(np.roll(A, (i,j), axis=(0,1)) for i in (-1,0,+1) for j in (-1,0,+1) if (i!=0 or j!=0))
  U = scipy.signal.convolve2d(A, K, mode='same', boundary='wrap')
  A = np.clip(A + growth(U), 0, 1)
  # img.set_array(A)
  return A

def compete_random(X,Y):
    Xnew = copy.deepcopy(X)
    Ynew = copy.deepcopy(Y)
    Xlarge = np.greater(X,Y) & Y>0
    Ylarge = np.greater(Y,X) & X>0
    XYequal = np.equal(Y,X) & X>0
    Xnew[Xlarge] = 3
    Ynew[Xlarge] = 0
    Ynew[Ylarge] = 3
    Xnew[Ylarge] = 0

    equalflat = np.flatnonzero(XYequal)
    equalrand = shuffle(equalflat)
    tos = int(len(equalrand)/2)
    xf = Xnew.flatten()
    yf = Ynew.flatten()
    xf[equalrand[:tos]] = 3
    yf[equalrand[:tos]] = 0
    yf[equalrand[tos:]] = 3
    xf[equalrand[tos:]] = 0
    Xnew = xf.reshape((size,size))
    Ynew = yf.reshape((size,size))
    return Xnew,Ynew

def compete_0(X,Y):
    Xnew = copy.deepcopy(X)
    Ynew = copy.deepcopy(Y)
    Xlarge = np.greater(X,Y) & Y>0
    Ylarge = np.greater(Y,X) & X>0
    XYequal = np.equal(Y,X) & X>0
    Xnew[Xlarge] = 3
    Ynew[Xlarge] = 0
    Ynew[Ylarge] = 3
    Xnew[Ylarge] = 0
    Ynew[XYequal] = 0
    Xnew[XYequal] = 0    
    return Xnew,Ynew

def update2(A,B,K,random = True):
  # global A
  ''' use convolution '''
  #U = sum(np.roll(A, (i,j), axis=(0,1)) for i in (-1,0,+1) for j in (-1,0,+1) if (i!=0 or j!=0))
  U = scipy.signal.convolve2d(A, K, mode='same', boundary='wrap')
  U2 = scipy.signal.convolve2d(B, K, mode='same', boundary='wrap')
  if random:
      Ub,U2b = compete_random(U,U2)
  else:
      Ub,U2b = compete_0(U,U2)
  A = np.clip(A + growth(Ub), 0, 1)
  B = np.clip(B + growth(U2b), 0, 1)
  # img.set_array(A)
  return A,B

#%%Initialization:
global size 
size = 128
gens = 200
rows = int(gens/5)

# Ao = create_circular_mask(size,size,radius=30,center=(int(size/2),int(size/2)))
# Bo = create_circular_mask(size,size,radius=20,center=(int(3*size/4),int(size/2)))

# Ao = create_square_mask(size,size,length=40,center=(int(size/4),int(size/2)))
# Bo = create_square_mask(size,size,length=60,center=(int(3*size/4),int(size/2)))

Ao = np.random.randint(2, size=(size, size))
Bo = np.random.randint(2, size=(size, size))

A = copy.deepcopy(Ao)
B = copy.deepcopy(Bo)


#%%Plotting scripts:
fig = plt.figure(dpi=150)
for g in range(1,gens+1):
    ax = fig.add_subplot(rows,5,g)
    rgb = np.dstack((A,B,np.zeros(np.shape(A))))
    plt.imshow(rgb)
    A,B = update2(A,B,K,random=True)
    plt.xlim([0,size])
    plt.ylim([0,size])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


A = copy.deepcopy(Ao)
B = copy.deepcopy(Bo)

fig = plt.figure(dpi=150)
for g in range(1,gens+1):
    ax = fig.add_subplot(rows,5,g)
    rgb = np.dstack((A,B,np.zeros(np.shape(A))))
    plt.imshow(rgb)
    A,B = update2(A,B,K,random=False)
    plt.xlim([0,size])
    plt.ylim([0,size])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


#%%Animation for Random boundary condition:
A = copy.deepcopy(Ao)
B = copy.deepcopy(Bo)
rgb_list = []
fig, ax = plt.subplots()
for g in range(1,gens+1):
    rgb = np.dstack((A,B,np.zeros(np.shape(A))))
    rgb_list.append(rgb)
    A,B = update2(A,B,K,random=True)
    
def animate(frame):
    # remove previous image
    ax.clear()    
    # get new image from list
    art = rgb_list[frame % len(rgb_list)]
    ax.set_title('Generation '+str(frame),fontsize=15,fontweight='bold')
    ax.set_xlim([0,size])
    ax.set_ylim([0,size])
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.imshow(art)
    return art

ani = animation.FuncAnimation(fig, animate,interval=50, 
                              frames = gens,blit=False, repeat_delay=1000,
                              cache_frame_data=False)
plt.show()

ani.save('random_v4_fast.mp4')


#%%Animation for 0-boundary condition:
A = copy.deepcopy(Ao)
B = copy.deepcopy(Bo)
rgb_list = []
fig, ax = plt.subplots()
for g in range(1,gens+1):
    rgb = np.dstack((A,B,np.zeros(np.shape(A))))
    rgb_list.append(rgb)
    A,B = update2(A,B,K,random=False)
    
def animate(frame):
    # remove previous image
    ax.clear()    
    # get new image from list
    art = rgb_list[frame % len(rgb_list)]
    ax.set_title('Generation '+str(frame),fontsize=15,fontweight='bold')
    ax.set_xlim([0,size])
    ax.set_ylim([0,size])
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.imshow(art)
    return art

ani = animation.FuncAnimation(fig, animate,interval=50, 
                              frames = gens,blit=False, repeat_delay=1000,
                              cache_frame_data=False)
plt.show()

ani.save('zero_v4_fast.mp4')