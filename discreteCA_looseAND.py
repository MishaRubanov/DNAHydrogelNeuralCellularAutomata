# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 19:14:30 2021

@author: rmish
"""


# You can change the color map you are using via the cmap keyword. The color map 'Greys' provides the effect you want. You can find a list of available maps on the scipy website.


#%%Functions and initalization:
import matplotlib.pyplot as plt
import numpy as np

def check_neighbor_AND(mat,xpos, ypos):
    xgate = 0
    ygate = 0
    totgate = 0
    if mat[xpos-1,ypos] == 1 and mat[xpos+1, ypos] == 1:
        xgate = 1    
    if mat[xpos,ypos-1] == 1 and mat[xpos,ypos+1] == 1:
        ygate = 1
    if xgate or ygate:
        totgate = 1
        # print(xpos,ypos)
    return totgate

def check_neighbor_strictAND(mat,xpos, ypos):
    xgate = 0
    ygate = 0
    totgate = 0
    if mat[xpos-1,ypos] == 1 and mat[xpos+1, ypos] == 1:
        xgate = 1    
    if mat[xpos,ypos-1] == 1 and mat[xpos,ypos+1] == 1:
        ygate = 1
    if xgate and ygate:
        totgate = 1
        # print(xpos,ypos)
    return totgate

def check_neighbor_looseAND(mat,xpos, ypos):
    xgate = 0
    ygate = 0
    totgate = 0
    if mat[xpos-1,ypos] == 1 and mat[xpos+1, ypos] == 1:
        xgate = 1    
    if mat[xpos,ypos-1] == 1 and mat[xpos,ypos+1] == 1:
        ygate = 1
    if xgate and ygate:
        totgate = 1
        # print(xpos,ypos)
    return totgate

def check_neighbor_XOR(mat,xpos, ypos):
    xgate = 0
    ygate = 0
    totgate = 0
    if mat[xpos-1,ypos]  != mat[xpos+1, ypos]:
        xgate = 1    
    if mat[xpos,ypos-1] != mat[xpos,ypos+1]:
        ygate = 1
    if xgate == 1 or ygate ==1 :
        totgate = 1
        # print(xpos,ypos)
    return totgate

def check_neighbor_OR(mat,xpos, ypos):
    xgate = 0
    ygate = 0
    totgate = 0
    if mat[xpos-1,ypos] == 1 or mat[xpos+1, ypos] == 1:
        xgate = 1    
    if mat[xpos,ypos-1] == 1 or mat[xpos,ypos+1] == 1:
        ygate = 1
    if xgate or ygate:
        totgate = 1
        # print(xpos,ypos)
    return totgate


#%%Initial nodes setup plot:
n = 7
x1 = np.zeros((n,n))
x2 = np.zeros((n,n))
x3 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i % 2 == 0 and j %2 == 0:
            x1[i,j] = 1 
        elif i % 2 == 1 and j % 2 == 1:
            x3[i,j] = 1
        elif i % 2 == 1 and j % 2 == 0:
            x2[i,j] = 1
        elif i % 2 == 0 and j % 2 == 1:
            x2[i,j] = 1
            
            
fig = plt.figure(dpi=150)
s1x,s1y = np.nonzero(x1)
s2x,s2y = np.nonzero(x2)
s3x,s3y = np.nonzero(x3)
plt.scatter(s1x,s1y,100,color='b',label='L1')
plt.scatter(s2x,s2y,color='r',label='L2')
plt.scatter(s3x,s3y,color='g',label='L3')
ax = fig.get_axes()[0]
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
# plt.legend(['L1','L2','L3'],Loc=)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15)
plt.tight_layout()
plt.savefig('hsetup_v1.png',format='png')

#%%2x2 matrix
fig = plt.figure(dpi=150)
gens = 7
n = 11
xinput = np.zeros((n,n))
for i in [4,6]:
    for j in [4,6]:
        xinput[i,j] = 1 
xgens = np.zeros((n,n,gens+1))
xgens[:,:,0] = xinput
# fig = plt.figure(dpi=150)

for g in range(1,gens+1):
    for i in range(1,n-1):
        for j in range(1,n-1):
            xgens[i,j,g]= check_neighbor_AND(xgens[:,:,g-1],i, j)
    ax = fig.add_subplot(3,gens,g)
    sx,sy = np.nonzero(xgens[:,:,g-1])
    plt.scatter(sx,sy,10,cmap='binary')
    plt.xlim([0,n])
    plt.ylim([0,n])
    ax.set_aspect('equal')
    ax.set_title('Gen '+str(g), fontfamily='serif', loc='left', fontsize='medium')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.setp(ax, ylabel='x axis label')


# plt.setp(axs[:, 0], ylabel='y axis label')("2x2 Matrix")
# ax.set_ylabel('2x2 matrix')# ,fontfamily='serif', loc='left', fontsize='medium')

#%%3x3 matrix
xinput = np.zeros((n,n))
for i in [2,4,6]:
    for j in [2,4,6]:
        xinput[i,j] = 1 
        
xgens = np.zeros((n,n,gens+1))
xgens[:,:,0] = xinput

for g in range(1,gens+1):
    for i in range(1,n-1):
        for j in range(1,n-1):
            xgens[i,j,g]= check_neighbor_AND(xgens[:,:,g-1],i, j)
    ax = fig.add_subplot(3,gens,g+gens)
    sx,sy = np.nonzero(xgens[:,:,g-1])
    plt.scatter(sx,sy,10,cmap='binary')
    plt.xlim([0,n])
    plt.ylim([0,n])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    


#%%4x4 matrix
xinput = np.zeros((n,n))
for i in [2,4,6,8]:
    for j in [2,4,6,8]:
        xinput[i,j] = 1 
xgens = np.zeros((n,n,gens+1))
xgens[:,:,0] = xinput
# fig = plt.figure(dpi=150)

for g in range(1,gens+1):
    for i in range(1,n-1):
        for j in range(1,n-1):
            xgens[i,j,g]= check_neighbor_AND(xgens[:,:,g-1],i, j)
    ax = fig.add_subplot(3,gens,g+gens*2)
    sx,sy = np.nonzero(xgens[:,:,g-1])
    plt.scatter(sx,sy,10,cmap='binary')
    plt.xlim([0,n])
    plt.ylim([0,n])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


#%%AND XOR switch: rect input
gens=15
n = 20
fig = plt.figure(dpi=150)
xinput = np.zeros((n,n))
for i in [8,10,12,14]:
    for j in [8,10,12,14]:
        xinput[i,j] = 1 
xgens = np.zeros((n,n,gens+1))
xgens[:,:,0] = xinput
for g in range(1,gens+1):
    for i in range(1,n-1):
        for j in range(1,n-1):
            if g%3 == 0:
                xgens[i,j,g]= check_neighbor_XOR(xgens[:,:,g-1],i, j)
            else:
                xgens[i,j,g]= check_neighbor_AND(xgens[:,:,g-1],i, j)
                
    ax = fig.add_subplot(3,5,g)
    sx,sy = np.nonzero(xgens[:,:,g-1])
    plt.scatter(sx,sy,10,cmap='binary')
    plt.xlim([0,n])
    plt.ylim([0,n])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


#%%AND XOR switch: triangle input
gens=15
n = 20
xinput = np.zeros((n,n))
for i in range(2,10):
    for j in range(2,10):
        xinput[i,j] = 1 

fig = plt.figure(dpi=150)
# xtput = np.tril(xinput,-1)
xgens = np.zeros((n,n,gens+1))
xgens[:,:,0] = xinput
for g in range(1,gens+1):
    for i in range(1,n-1):
        for j in range(1,n-1):
            if g%2 == 0:# or g%4 == 1:
                xgens[i,j,g]= check_neighbor_strictAND(xgens[:,:,g-1],i, j)
            # elif g%4 == 3:                
            #     xgens[i,j,g]= check_neighbor_OR(xgens[:,:,g-1],i, j)
            else:
                xgens[i,j,g]= check_neighbor_XOR(xgens[:,:,g-1],i, j)
                
    ax = fig.add_subplot(3,5,g)
    sx,sy = np.nonzero(xgens[:,:,g-1])
    plt.scatter(sx,sy,10,cmap='binary')
    plt.xlim([0,n])
    plt.ylim([0,n])
    ax.set_aspect('equal')
    # plt.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


