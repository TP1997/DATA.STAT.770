import numpy as np
from numpy.linalg import norm

#%% a)
dims = [2,3,5,7,10,13,17]
N = 10000000
print('Proportion of points inside hypersphere:')
for dim in dims:
    points = np.random.uniform(low=-1, high=1, size=(N,dim))
    distances = norm(points, axis=1)
    n_inside = np.sum(distances <= 1)
    print('At dimension {} : {}/{} = {}'.format(dim, n_inside, N, n_inside/N))
    
#%% b)
print('Proportion of points inside the shell of hypersphere:')
for dim in dims:
    points = np.random.uniform(low=-1, high=1, size=(N,dim))
    distances = norm(points, axis=1)
    
    inside_hs_mask = distances <= 1
    n_inside_hs = np.sum(inside_hs_mask)
    
    distances = distances[inside_hs_mask]
    inside_shell_mask = distances >= 0.95
    n_inside_shell = np.sum(inside_shell_mask)
    
    print('At dimension {} : {}/{} = {}'.format(dim, n_inside_shell, n_inside_hs, n_inside_shell/n_inside_hs))

