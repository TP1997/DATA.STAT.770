#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 04:21:47 2021

@author: tuomas
"""
import numpy as np
import os
from sklearn.manifold import SpectralEmbedding, Isomap, LocallyLinearEmbedding
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import norm

#%%
os.chdir('/home/tuomas/Python/DATA.STAT.770/E12')
data = np.loadtxt('swissroll.dat')

c = norm(data[:,[0,1]]-np.array([0.5,0.5]), axis=1)
c2 = np.cos(data[:,0]) + np.sin(data[:,1])# + np.cos(data[:,2]-120)
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hsv")
ax.scatter(data[:,0],data[:,1],data[:,2], c=c2, cmap=cmhot)
ax.view_init(elev=70., azim=35)