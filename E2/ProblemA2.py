import numpy as np
import os

#%%
os.chdir('/home/tuomas/Python/DATA.STAT.770/E2/')
data = np.loadtxt('noisy_sculpt_faces.txt')

images = data[:,:-3]
angles_gt = data[:,-3:]

#%% a) Nearest neighbor predictor & errors
from numpy.linalg import norm

def NN1_predictor(images, angles_gt):
    #angles_pose_pred = []
    #angles_light_pred = []
    angles_pred = []
    for i in range(images.shape[0]):
        img = images[i]
        distances = np.square(np.sum(images - img, axis=1))
        distances[i] = np.finfo('float').max
        closestidx = np.argmin(distances)
        
        angles_pred.append(angles_gt[closestidx])
        #angles_pose_pred.append(data[closestidx, 256:258])
        #angles_light_pred.append(data[closestidx, -1])
        
    return np.array(angles_pred)
        
def leaveoneout_error(angles_gt, angles_pred):
    errors = np.sum(angles_gt - angles_pred, axis=1)
    errors = np.square(errors)
    
    return errors.sum()

#%% 2)
def forward_selection_new(images, angles_gt):
    errors_iter = []
    n = 1
    best_features = []
    while True:
        errors = []
        # Calculate error terms with specific set of features
        for i in range(images.shape[1]):
            if i in best_features: 
                errors.append(np.finfo('float').max)
                continue

            curr_features = [i]
            imgs = images[:, curr_features]
            angles_pred = NN1_predictor(imgs, angles_gt)
            error = leaveoneout_error(angles_gt, angles_pred)
            errors.append(error)
            
        errors = np.array(errors)
        bf = np.argmin(errors)
        # If there are unused features
        if len(best_features) < images.shape[1]:
            best_features.append(bf)
            
            imgs = images[:, best_features]
            angles_pred = NN1_predictor(imgs, angles_gt)
            curr_error_tot = leaveoneout_error(angles_gt, angles_pred)
            
            errors_iter.append(curr_error_tot)
            #images = np.delete(images, bf, axis=1)
            
        else:
            break
        
        print('Round {}:'.format(n))
        print('Error = {}'.format(errors_iter[-1]))
        print('Best features = {}'.format(best_features))
        n+=1
        
    return np.array(best_features), np.array(errors_iter[:])

bf, errors_itr = forward_selection_new(images, angles_gt)

#%% Some plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(11,8))
plt.title('Preformance')
plt.ylabel('SSE')
plt.xlabel('Features')
plt.plot(np.arange(1,bf.shape[0]+1), errors_itr, '.-', color='red')




