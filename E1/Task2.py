import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def generate_data(N, dim):
    xs = np.random.multivariate_normal(np.zeros(dim), np.diag(np.ones(dim)), size=N)
    # Target
    ys = xs[:,0] + np.sin(5*xs[:,0])
    
    trainX = xs[:1000]
    trainY = ys[:1000]
    testX = xs[1000:]
    testY = ys[1000:]
    
    return trainX, trainY, testX, testY

def fiveNN(trainX, trainY, testX):
    predY = []
    for tstx in testX:
        distances = norm(tstx-trainX, axis=1)
        idxs = np.argpartition(distances, 5)[:5]
        py = np.mean(trainY[idxs])
        predY.append(py)
        
    return predY

def pred_error_mse(testY, predY):
    sqdiff = (testY-predY)**2
    
    return np.mean(sqdiff)

def plot(x1, testY, predY, n, dim, mspe):
    plt.figure(n)
    plt.title('Dimension {}, MSPE={}'.format(dim, np.round(mspe,3)))
    tv, = plt.plot(x1, testY, '.', label='Target variable')
    pv, = plt.plot(x1, predY, '.', label='Predicted value')
    plt.xlabel('x1')
    plt.legend(handles=[tv,pv])
    
    return
    

dims = [2,3,5,7,10,13,17]
N = 2000
for n in range(len(dims)):
    d = dims[n]
    print('Dimension {}'.format(d))
    trainX, trainY, testX, testY = generate_data(N, d)
    predY = fiveNN(trainX, trainY, testX)
    
    # Mean-squared prediction error
    mspe = pred_error_mse(testY, predY)
    # Plotting
    plot(testX[:,0], testY, predY, n, d, mspe)
    
    






