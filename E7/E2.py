import numpy as np
import os

#%%
os.chdir('/home/tuomas/Python/DATA.STAT.770/E7')
data = np.loadtxt('swissroll.dat')

#%%
limits = np.arange(0,1.2,0.2)
cubes = np.zeros(1000)


for i in range(1000):
    s = data[i]
    posx = limits-s[0]
    posy = limits-s[1]
    posz = limits-s[2]
    
   # print(posx)
    #print(np.nonzero(posx>0))
    x = np.nonzero(posx>0)[0][0]-1
    y = np.nonzero(posy>0)[0][0]-1
    z = np.nonzero(posz>0)[0][0]-1
    
    cubes[i] = x+5*(y+5*z)
    
#%%



#%%
x=np.array([-0.57337789, -0.37337789, -0.17337789,  0.02662211,  0.22662211,  0.42662211])
res=np.nonzero(x>0)[0][0]

#%%
c = [[[]]]*9
x=0
y=0
z=0
c[x+3*(y+3*z)].append(0)