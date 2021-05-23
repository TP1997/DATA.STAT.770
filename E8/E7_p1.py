#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 02:10:55 2021

@author: tuomas
"""
import numpy as np
import os
from sklearn.manifold import SpectralEmbedding, Isomap, LocallyLinearEmbedding
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import norm

#%%
os.chdir('/home/tuomas/Python/DATA.STAT.770/E7')
data = np.loadtxt('swissroll.dat')

c = norm(data[:,[0,1]]-np.array([0.5,0.5]), axis=1)
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hsv")
ax.scatter(data[:,0],data[:,1],data[:,2], c=c, cmap=cmhot)
ax.view_init(elev=70., azim=35)
#%% Laplacian Eigenmap method
le_embedding = SpectralEmbedding(n_components=2)
data_proj1 = le_embedding.fit_transform(data)

#%% Isomap method
isomap_embedding = Isomap(n_components=2)
data_proj2 = isomap_embedding.fit_transform(data)

#%% Locally Linear Embedding method
ll_embedding = LocallyLinearEmbedding(n_components=2)
data_proj3 = ll_embedding.fit_transform(data)

#%% Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,15))
ax1.scatter(data_proj1[:,0], data_proj1[:,1], c=c, cmap=cmhot)
ax1.title.set_text('Laplacian Eigenmap method')
ax2.scatter(data_proj2[:,0], data_proj2[:,1],c=c, cmap=cmhot)
ax2.title.set_text('Isomap method')
ax3.scatter(data_proj3[:,0], data_proj3[:,1],c=c, cmap=cmhot)
ax3.title.set_text('Locally Linear Embedding method')