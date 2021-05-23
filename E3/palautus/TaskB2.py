import numpy as np
import os
from numpy.linalg import eig
import matplotlib.pyplot as plt
import itertools

#%% Load the data
os.chdir('/home/tuomas/Python/DATA.STAT.770/E3')
data_rw = np.loadtxt('winequality-red.txt')
data_ww = np.loadtxt('winequality-white.txt')

mean_rw = np.mean(data_rw, axis=0)
mean_ww = np.mean(data_ww, axis=0)

data_rw_cent = data_rw - mean_rw
data_ww_cent = data_ww - mean_ww
#%% Find principal components for red wines
rw_cov = np.cov(data_rw_cent, rowvar=False)
n = 2
rw_eig = eig(rw_cov)
W = rw_eig[1][:n]

rw_projected = data_rw_cent@W.T

#%% Compute the amount of variance explained
pov_rw = np.sum(rw_eig[0][:n]) / np.sum(rw_eig[0])

#%% Plot the projected data
rw_quality = (data_rw[:,-1]-5).astype('int')
rw_quality[rw_quality>2] = 2

rw_projected_cat = []
rw_projected_cat.append(rw_projected[rw_quality==0])
rw_projected_cat.append(rw_projected[rw_quality==1])
rw_projected_cat.append(rw_projected[rw_quality==2])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(rw_projected_cat[0][:,0],rw_projected_cat[0][:,1], c='r', label='low')
ax.scatter(rw_projected_cat[1][:,0],rw_projected_cat[1][:,1], c='g', label='medium')
ax.scatter(rw_projected_cat[2][:,0],rw_projected_cat[2][:,1], c='b', label='high')

ax.legend(title='Quality')
ax.set_title('''Projection of red wine data into two first principal components
             \n Amount of variance explained: {}'''.format(np.round(pov_rw, 3)))

#%% Find principal components for white wines
ww_cov = np.cov(data_ww_cent, rowvar=False)
n = 2
ww_eig = eig(ww_cov)
W = ww_eig[1][:n]
ww_projected = data_ww_cent@W.T

#%% Compute the amount of variance explained
pov_ww = np.sum(ww_eig[0][:n]) / np.sum(ww_eig[0])

#%% Plot the projected data
ww_quality = (data_ww[:,-1]-5).astype('int')
ww_quality[ww_quality>2] = 2


ww_projected_cat = []
ww_projected_cat.append(ww_projected[ww_quality==0])
ww_projected_cat.append(ww_projected[ww_quality==1])
ww_projected_cat.append(ww_projected[ww_quality==2])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ww_projected_cat[0][:,0],ww_projected_cat[0][:,1], c='r', label='low')
ax.scatter(ww_projected_cat[1][:,0],ww_projected_cat[1][:,1], c='g', label='medium')
ax.scatter(ww_projected_cat[2][:,0],ww_projected_cat[2][:,1], c='b', label='high')

ax.legend(title='Quality')
ax.set_title('''Projection of white wine data into two first principal components
             \n Amount of variance explained: {}'''.format(np.round(pov_ww, 3)))

#%% Varaible ranking based on variance
ww_vars = np.diag(ww_cov)
top2 = np.argsort(ww_vars)[-n:][::-1]

data_ww_varrank = data_ww_cent[:,top2]

ww_ranked_cat = []
ww_ranked_cat.append(data_ww_varrank[ww_quality==0])
ww_ranked_cat.append(data_ww_varrank[ww_quality==1])
ww_ranked_cat.append(data_ww_varrank[ww_quality==2])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ww_ranked_cat[0][:,0],ww_ranked_cat[0][:,1], c='r', label='low')
ax.scatter(ww_ranked_cat[1][:,0],ww_ranked_cat[1][:,1], c='g', label='medium')
ax.scatter(ww_ranked_cat[2][:,0],ww_ranked_cat[2][:,1], c='b', label='high')

ax.legend(title='Quality')
ax.set_title('White wine data using variable ranking.')











