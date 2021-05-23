import numpy as np
from metric_learn import NCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#%%
X, y = load_iris(return_X_y=True)
trainX, testX, trainY, testY = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.7)
nca = NCA(max_iter=500, n_components=4)
nca.fit(trainX, trainY)
M = nca.get_mahalanobis_matrix()

#%% 1nn-classification
from numpy.linalg import norm

def Mahalanobis_dist(tstX, trainX, M):
    diff = tstx-trainX
    dists = [d@M@d.T for d in diff]
    return dists

def Euclidean_dist(tstx, trainX):
    return norm(tstx-trainX, axis=1)


predY_euclidean = []
for tstx in testX:
    distances = Euclidean_dist(tstx, trainX)
    idx = np.argmin(distances)
    predy = trainY[idx]
    predY_euclidean.append(predy)
    
predY_euclidean = np.array(predY_euclidean)
euclidean_acc = np.sum(predY_euclidean==testY)
print('Nearest neighbor classification using euclidean distance:')
print('accuracy = {}/{} = {}'.format(euclidean_acc, testY.shape[0], np.round(euclidean_acc/testY.shape[0],4)))

predY_mahalanobis = []
for tstx in testX:
    distances = Mahalanobis_dist(tstx, trainX, M)
    idx = np.argmin(distances)
    predy = trainY[idx]
    predY_mahalanobis.append(predy)
    
predY_mahalanobis = np.array(predY_mahalanobis)
mahalanobis_acc = np.sum(predY_mahalanobis==testY)
print('Nearest neighbor classification using mahalanobis distance:')
print('accuracy = {}/{} = {}'.format(mahalanobis_acc, testY.shape[0], np.round(mahalanobis_acc/testY.shape[0],4)))


