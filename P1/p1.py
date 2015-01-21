from pylab import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os

os.chdir('/Users/sangahhan/Workspace/School/CSC320/P1/images/')


###################
#Run the following for grayscale images to show up correctly and for images
#to display immediately
#%matplotlib
#gray()
########################

def my_norm(v):
    return np.sqrt(np.sum(v**2))
    
def displaced_copies(g,f,m_n):
    ''' Return copies of g & f such that f is displaced onto g m down, n left'''
    
    m,n = m_n
    
    M = abs(m)
    N = abs(n)
    
    if m < 0:
        if n < 0: # m is neg, n is neg
            g_copy = g[:-M, :-N]
            f_copy = f[M:, N:]
        elif n > 0: # m is neg, n is pos
            g_copy = g[:-M, N:]
            f_copy = f[M:, :-N]
        else: # m is neg, n is 0
            g_copy = g[:-M, :]
            f_copy = f[M:, :]
    elif m > 0: 
        if n < 0: # m is pos, n is neg
            g_copy = g[M:, :-N]
            f_copy = f[:-M, N:] 
        elif n > 0: # m is pos, n is pos
            g_copy = g[M:, N:]
            f_copy = f[:-M, :-N]
        else: # m is pos, n is 0
            g_copy = g[M:, :]
            f_copy = f[:-M, :]
    else:
        if n < 0: # m is 0, n is neg
            g_copy = g[:, :-N]
            f_copy = f[:, N:]
        elif n > 0: # m is 0, n is pos
            g_copy = g[:, N:]
            f_copy = f[:, :-N]
        else: # m is 0, n is 0
            g_copy = g
            f_copy = f
    
    return (g_copy, f_copy)
        

def ncc(g,f,m_n):
    
    # precondition: g.shape >= f.shape
    g_copy, f_copy = displaced_copies(g, f, m_n)
    
    result = np.dot(g_copy.flatten(),f_copy.flatten()) / (my_norm(g_copy) * my_norm(f_copy))
    
    #cost = (np.sum((img1 - img1.mean())*(img2 - img2.mean())) / 
    #       (np.sqrt(np.sum((img1 - img1.mean())**2)) * 
    #        np.sqrt(np.sum((img2 - img2.mean())**2))))
    
    #mult = (g*f[m:,n:])
    #result = mult.sum()
   
    return result


def ssd(g,f,m_n):
    g_copy, f_copy = displaced_copies(g, f, m_n)
    
    diff = np.subtract(g_copy,f_copy)
    return np.sum(diff * diff)
    
    
def get_displacement_vectors(n):
    result = []
    for i in range(n+1):
        for j in range(n+1):
            if (i,j) not in result:
                result.append((i,j))
            if (-i,j) not in result:
                result.append((-i,j))
            if (i,-j) not in result:
                result.append((i,-j))
            if (-i,-j) not in result:
                result.append((-i,-j))
    return result        


def get_scores(func, g, f, displacement_vectors):
    results = {}
    for v in displacement_vectors:
        results[func(g, f, v)] = v
    
    return results
    
    
def best_match(func, g, f, displacement_vectors):
    results = get_scores(func, g, f, displacement_vectors)
    if func is ssd:
        return results[min(results)]
    return results[max(results)]
 
i = imread('00757v.jpg')
i.astype(uint8)

# crop borders
w = i.shape[1]
#w_5 = np.ceil(w * .05)
#i = i[w_5:-w_5, w_5:-w_5]
#normalize image to be between 0 and 1
i *= 255.0/i.max() 


# cut image into three peices
l = i.shape[0]
b = i[:(l/3)]
g = i[(l/3):((l/3)*2)]
r = i[((l/3)*2):l-(l%3)]


x = zeros(g.shape + (3,)).astype(uint8)
x[:,:,0] = r
x[:,:,1] = g
x[:,:,2] = b
figure(); imshow(x)

displacements = get_displacement_vectors(10)

g_match_ssd = best_match(ssd, b, g, displacements)
r_match_ssd = best_match(ssd, b, r, displacements)

new_g = displaced_copies(b, g, g_match_ssd)
new_r = displaced_copies(b, r, r_match_ssd)

y = zeros(new_r[0].shape + (3,)).astype(uint8)
y[:,:,0] = new_r[1]
y[:,:,2] = new_r[0]
figure(); imshow(y)

z = zeros(new_g[0].shape + (3,)).astype(uint8)
z[:,:,0] = new_g[1]
z[:,:,1] = new_g[0]
figure(); imshow(z)


