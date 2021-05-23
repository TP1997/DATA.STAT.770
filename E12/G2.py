import numpy as np
from numpy.linalg import norm
import os
os.chdir('/home/tuomas/Python/DATA.STAT.770/E12')

#%%
# Get true neighbourdood samples
def get_neighborhoods(data):
    N = 5+1
    distances = [norm(d-data, axis=1) for d in data]
    neighbourhoods = np.array([np.argpartition(d, N)[:N] for d in distances])
    
    return neighbourhoods

# Calculate precision and recall values    
def calc_pr(data_proj, neigh_orig):
    N=5
    avg_precisions = []
    avg_recalls = []
    ts = []
    
    T = 1.0
    while T > 1e-4:
        print('T = {}'.format(np.round(T,5)))
        # Get projected neighbourhoods
        proj_distances = [np.log(norm(dp-data_proj, axis=1)) for dp in data_proj]
        neigh_proj = np.array([np.argwhere(pd <= np.log(T)).ravel() for pd in proj_distances])
        
        # Get neighbourhood intersections between projected and original samples
        neigh_intersections = [np.intersect1d(neo,nep) for neo,nep in zip(neigh_orig, neigh_proj)]
        
        # Calculate average precision and recall
        avg_prec = np.mean([(num.shape[0]-1) / (denom.shape[0]-1) if denom.shape[0]>1 else 0 for num, denom in zip(neigh_intersections, neigh_proj)])
        avg_rec = np.mean([(num.shape[0]-1) / N for num in neigh_intersections])
        avg_precisions.append(avg_prec)
        avg_recalls.append(avg_rec)
        
        ts.append(T)
        T *= 9e-1
        
    return (avg_precisions, avg_recalls, ts)

# Calculate precision and recall values    
def calc_pr2(data_proj, neigh_orig):
    N=5
    avg_precisions = []
    avg_recalls = []
    ts = []
    
    T = 1.0
    for T in np.arange(50, 1.0, -0.1):
        print('T = {}'.format(np.round(T,5)))
        # Get projected neighbourhoods
        proj_distances = [np.log(norm(dp-data_proj, axis=1)) for dp in data_proj]
        neigh_proj = np.array([np.argwhere(pd <= np.log(T)).ravel() for pd in proj_distances])
        
        # Get neighbourhood intersections between projected and original samples
        neigh_intersections = [np.intersect1d(neo,nep) for neo,nep in zip(neigh_orig, neigh_proj)]
        
        # Calculate average precision and recall
        avg_prec = np.mean([(num.shape[0]-1) / (denom.shape[0]-1) if denom.shape[0]>1 else 0 for num, denom in zip(neigh_intersections, neigh_proj)])
        avg_rec = np.mean([(num.shape[0]-1) / N for num in neigh_intersections])
        avg_precisions.append(avg_prec)
        avg_recalls.append(avg_rec)
        
        ts.append(T)
        
    return (avg_precisions, avg_recalls, ts)
    
#%%
# Original data
data_sr = np.loadtxt('swissroll.dat')
data_hs = np.loadtxt('halfsphere.dat')
data_ckd = np.loadtxt('ckd.dat')

# Get neighbourhoods for original data
neigh_orig_sr = get_neighborhoods(data_sr)
neigh_orig_hs = get_neighborhoods(data_hs)
neigh_orig_ckd = get_neighborhoods(data_ckd)

# NeRV data
os.chdir('nervs')
# lambda=0.1 and lambda=0.8
data_sr_proj01 = np.loadtxt('swissroll_out01.dat')
data_sr_proj08 = np.loadtxt('swissroll_out08.dat')
data_hs_proj01 = np.loadtxt('halfsphere_out01.dat')
data_hs_proj08 = np.loadtxt('halfsphere_out08.dat')
data_cdk_proj01 = np.loadtxt('ckd_out01.dat')
data_cdk_proj08 = np.loadtxt('ckd_out08.dat')

#%% Get precision & recall values for NeRVs
avgp_sr01, avgr_sr01, ts = calc_pr(data_sr_proj01, neigh_orig_sr)
avgp_sr08, avgr_sr08, _ = calc_pr(data_sr_proj08, neigh_orig_sr)

avgp_hs01, avgr_hs01, _ = calc_pr(data_hs_proj01, neigh_orig_hs)
avgp_hs08, avgr_hs08, _ = calc_pr(data_hs_proj08, neigh_orig_hs)

avgp_ckd01, avgr_ckd01, _ = calc_pr(data_cdk_proj01, neigh_orig_ckd)
avgp_ckd08, avgr_ckd08, _ = calc_pr(data_cdk_proj08, neigh_orig_ckd)

avgp_ckd012, avgr_ckd012, _ = calc_pr2(data_cdk_proj01, neigh_orig_ckd)
avgp_ckd082, avgr_ckd082, _ = calc_pr2(data_cdk_proj08, neigh_orig_ckd)

#%% Get tSNE projections
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
data_sr_projTSNE = tsne.fit_transform(data_sr)
data_hs_projTSNE = tsne.fit_transform(data_hs)
data_ckd_projTSNE = tsne.fit_transform(data_ckd)

#%% Get precision & recall values for TSNEs
avgp_srTSNE, avgr_srTSNE, _ = calc_pr(data_sr_projTSNE, neigh_orig_sr)
avgp_srTSNE2, avgr_srTSNE2, _ = calc_pr2(data_sr_projTSNE, neigh_orig_sr)
avgp_hsTSNE, avgr_hsTSNE, _ = calc_pr(data_hs_projTSNE, neigh_orig_hs)
avgp_hsTSNE2, avgr_hsTSNE2, _ = calc_pr2(data_hs_projTSNE, neigh_orig_hs)
avgp_ckdTSNE, avgr_ckdTSNE, _ = calc_pr(data_ckd_projTSNE, neigh_orig_ckd)
avgp_ckdTSNE2, avgr_ckdTSNE2, _ = calc_pr2(data_ckd_projTSNE, neigh_orig_ckd)


#%% Plot the results
import matplotlib.pyplot as plt

def plotNeRV(rec01, pre01, rec08, pre08, title, l=1e-4, h=1):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    sr01 = ax1.scatter(rec01, pre01, c='r')#cmap='Reds', c=np.arange(10,1,step=-9/88))
    sr08 = ax1.scatter(rec08, pre08, c='b')
    ax1.set_xlabel('Average recall')
    ax1.set_ylabel('Average precision')

    ax2 = ax1.secondary_xaxis('top')
    ax2.set_xlim(l, h)
    ax2.set_ticks([l, h])
    ax2.set_xlabel('T')

    plt.legend([sr01, sr08], ['lambda=0.1', 'lambda=0.8'], loc='lower left')
    plt.title(title)


plotNeRV(avgr_sr01, avgp_sr01, avgr_sr08, avgp_sr08, 'NeRV - Swissroll')
plotNeRV(avgr_hs01, avgp_hs01, avgr_hs08, avgp_hs08, 'NeRV - Halfsphere')
plotNeRV(avgr_ckd01, avgp_ckd01, avgr_ckd08, avgp_ckd08, 'NeRV - Chronic Kidney Disease')
plotNeRV(avgr_ckd012, avgp_ckd012, avgr_ckd082, avgp_ckd082, 'NeRV - Chronic Kidney Disease 2', l=1, h=10)

#%%
# Swissroll - Halfsphere - CKD - TSNE
fig = plt.figure()
ax1 = fig.add_subplot(111)

srtsne = ax1.scatter(avgr_srTSNE, avgp_srTSNE, c='r')#cmap='Reds', c=np.arange(10,1,step=-9/88))
hstsne = ax1.scatter(avgr_hsTSNE, avgp_hsTSNE, c='b')
ckdtsne = ax1.scatter(avgr_ckdTSNE, avgp_ckdTSNE, c='g')
plt.xlabel('Average recall')
plt.ylabel('Average precision')

ax2 = ax1.secondary_xaxis('top')
ax2.set_xlim(1e-4, 1)
ax2.set_ticks([1e-4, 1])
ax2.set_xlabel('T')

plt.legend([srtsne, hstsne, ckdtsne], ['swissroll','halfsphere','ckd'], loc='lower left')
plt.title('TSNE')
    
#%%
# Swissroll - Halfsphere - CKD - TSNE
fig = plt.figure()
ax1 = fig.add_subplot(111)

srtsne = ax1.scatter(avgr_srTSNE2, avgp_srTSNE2, c='r')#cmap='Reds', c=np.arange(10,1,step=-9/88))
hstsne = ax1.scatter(avgr_hsTSNE2, avgp_hsTSNE2, c='b')
ckdtsne = ax1.scatter(avgr_ckdTSNE2, avgp_ckdTSNE2, c='g')
plt.xlabel('Average recall')
plt.ylabel('Average precision')

ax2 = ax1.secondary_xaxis('top')
ax2.set_xlim(1.1, 50)
ax2.set_ticks(np.arange(50, 1.0, -0.1))
ax2.set_xlabel('T')

plt.legend([srtsne, hstsne, ckdtsne], ['swissroll','halfsphere','ckd'], loc='lower left')
plt.title('TSNE 2')

#%%
plt.legend([srtsne, hstsne, ckdtsne], ['swissroll','halfsphere','ckd'], loc='lower left')
plt.title('TSNE')