import numpy as np
import os
from sklearn.manifold import SpectralEmbedding, Isomap
#%%
os.chdir('/home/tuomas/Python/DATA.STAT.770/E7')
data = np.loadtxt('swissroll.dat')

#%% Caclulate true neighbourdood samples
from numpy.linalg import norm
N = 5+1
distances = [norm(d-data, axis=1) for d in data]
neighbourhoods = np.array([np.argpartition(d, N)[:N] for d in distances])

#%% Laplacian Eigenmap method
le_embedding = SpectralEmbedding(n_components=2)
data_proj1 = le_embedding.fit_transform(data)

#%% Isomap method
isomap_embedding = Isomap(n_components=2)
data_proj2 = isomap_embedding.fit_transform(data)

#%% Calculate precison & recall over various T
T = 1.0
le_aps = []
le_ars = []
im_aps = []
im_ars = []
ts = []
while T > 1e-4:
    print('T = {}'.format(np.round(T,5)))
    # Calculate projected neighbourhoods
    le_distances = [np.log(norm(dp1-data_proj1, axis=1)) for dp1 in data_proj1]
    le_neighbourhoods = np.array([np.argwhere(led <= np.log(T)).ravel() for led in le_distances])
    im_distances = [np.log(norm(dp2-data_proj2, axis=1)) for dp2 in data_proj2]
    im_neighbourhoods = np.array([np.argwhere(imd <= np.log(T)).ravel() for imd in im_distances])
    # Calculate intersections
    le_inters = [np.intersect1d(n,r) for n,r in zip(neighbourhoods, le_neighbourhoods)]
    im_inters = [np.intersect1d(n,r) for n,r in zip(neighbourhoods, im_neighbourhoods)]
    # Calculate average precision and recall
    le_ap = np.mean([(num.shape[0]-1) / (denom.shape[0]-1) if denom.shape[0]>1 else 0 for num, denom in zip(le_inters, le_neighbourhoods)])
    im_ap = np.mean([(num.shape[0]-1) / (denom.shape[0]-1) if denom.shape[0]>1 else 0 for num, denom in zip(im_inters, im_neighbourhoods)])
    le_ar = np.mean([(num.shape[0]-1) / (N-1) for num in le_inters])
    im_ar = np.mean([(num.shape[0]-1) / (N-1) for num in im_inters])
    
    le_aps.append(le_ap)
    im_aps.append(im_ap)
    le_ars.append(le_ar)
    im_ars.append(im_ar)
    
    ts.append(T)
    T *= 9e-1
   
#%% Plot the results
import matplotlib.pyplot as plt
le = plt.scatter(le_ars, le_aps, cmap='Reds', c=np.arange(10,1,step=-9/88))
im = plt.scatter(im_ars, im_aps, cmap='Reds', c=np.arange(10,1,step=-9/88))
plt.xlabel('Average recall')
plt.ylabel('Average precision')
plt.legend([le, im], ['Laplacian Eigenmap', 'Isomap'], colors=['b','r'])
