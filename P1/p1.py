from pylab import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os


###################
#Run the following for grayscale images to show up correctly and for images
#to display immediately
#%matplotlib
#gray()
########################
    
def displaced_copies(g,f,m_n):
    ''' Return copies of g & f st they only contain the intersection of g and f 
    when f is displaced m down and n left.
    
    Keyword arguments:
    g -- base image
    f -- patch 
    m_n -- a tuple that contains the values (m,n).
    '''
    
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
    ''' Return the normalized cross correlation of g and f, where f is displaced
    m down and n left.
    
    Keyword arguments:
    g -- base image
    f -- patch 
    m_n -- a tuple that contains the values (m,n).
    '''

    g_copy, f_copy = displaced_copies(g, f, m_n)
    
    G = g_copy - g.mean()
    F = f_copy - f.mean()
    
    subtract_mean = (G * F).sum()
    std_devs = np.sqrt((G*G).sum() * (F*F).sum())
    
    return subtract_mean / std_devs
    
    
def zero_mean_ncc(g,f,m_n):
    ''' Return the zero-mean normalized cross correlation of g and f, where f is 
    displaced m down and n left.
    
    Note: 
        This function is not used. I found that this tended to be really 
        inaccurate compared to non-zero-mean NCC.
    
    Keyword arguments:
    g -- base image
    f -- patch 
    m_n -- a tuple that contains the values (m,n).
    '''

    g_copy, f_copy = displaced_copies(g, f, m_n)
    
    dot = np.dot(g_copy.flatten(), f_copy.flatten()).sum()
    norm = np.linalg.norm(g_copy) * np.linalg.norm(f_copy)
    
    return dot / norm


def ssd(g,f,m_n):
    ''' Return the sum of squared differences for g and f, where f is displaced
    m down and n left.
    
    Keyword arguments:
    g -- base image
    f -- patch 
    m_n -- a tuple that contains the values (m,n).
    '''

    g_copy, f_copy = displaced_copies(g, f, m_n)
    
    diff = (g_copy.astype("float") - f_copy.astype("float"))

    return np.sum(diff * diff)
    
    
def get_displacement_vectors(n, step=1):
    ''' Return a list of vectors (represented by tuples) that represent possible
    displacement points that go up to a certain range.
    
    Keyword arguments:
    n -- range for the displacement points -> [-n, n]
    '''
    result = [(0,0)]
    for i in range(1, n+1, step):
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
    ''' Return a dictionary that maps scores to their corresponding displacement
    vector.
    
    Keyword arguments:
    func -- the function to compute the score with
    g -- base image
    f -- patch 
    displacement_vectors -- a list of displacement vectors to try
    '''
    
    results = {}
    for v in displacement_vectors:
        score = func(g, f, v)
        if score not in results:
            results[score] = v
    
    return results
    
    
def best_match(func, g, f, displacement_vectors):
    ''' Return the displacement vector with the ideal score.
    
    Keyword arguments:
    func -- the function to compute the score with
    g -- base image
    f -- patch 
    displacement_vectors -- a list of displacement vectors to try
    '''
    
    results = get_scores(func, g, f, displacement_vectors)
    if func is ssd:
        result = results[min(results)]
    else:
        result = results[max(results)]
    
    return result
 
 
def shift(img, displacement):
    ''' Return an image that has been displaced a certain amount, with its 
    remaining space filled with zeros (black).
    
    Keyword arguments:
    img -- the image to displace
    displacement -- the displacement vector by which we will displace the image
    '''
    
    m,n = displacement
    M = abs(m)
    N = abs(n)
    if m > 0:
        if n > 0:
            result = np.lib.pad(img[:-M,:-N],((M, 0),(N, 0)),mode='constant')
        else:
            result = np.lib.pad(img[:-M,N:],((M, 0),(0, N)),mode='constant')
    else:
        if n > 0:
            result = np.lib.pad(img[M:,:-N],((0, M),(N, 0)),mode='constant')
        else:
            result = np.lib.pad(img[M:,N:],((0, M),(0, N)),mode='constant')
    return result
    
    
def crop(img, displacement):
    ''' Return an image that has been cropped based on its displacement.
    
    Keyword arguments:
    img -- the image to crop
    displacement -- the displacement vector by which we will crop the image
    '''
    
    m,n = displacement
    M = abs(m)
    N = abs(n)
    if m < 0:
        if n < 0:
            result = img[:-M,:-N]
        else:
            result = img[:-M,N:]
    else:
        if n < 0:
            result = img[M:,:-N]
        else:
            result = img[M:,N:]
    return result
    
    
def max_displacement(v1, v2):
    ''' Return the maximum (wrt each axis) displacement based on absolute value, 
    given two displacements
    '''
    
    m1, n1 = v1
    m2, n2 = v2
    
    max_m = max([abs(m1), abs(m2)])
    max_n = max([abs(n1), abs(n2)])
    
    if max_m == abs(m1):
        if max_n == abs(n1):
            result = (m1, n1)
        else:
            result = (m1, n2)
    else:
        if max_n == abs(n1):
            result = (m2, n1)
        else:
            result = (m2, n2)
    
    return result
    

def normalize_image(img):
    ''' Normalize the given image to be in the range 0 & 255. '''
    
    img *= (255.0/img.max())


def part1(img_name, func, displacement_range=10):
    ''' Return an image that has been combined to be in full colour format from 
    the three-channel input image that was given.
    
    Keyword arguments:
    func -- corresponds to a function that implements some sort of patch-
            matching algorithm
    img_name -- name of the file with the 3-channelled photo (in the style of
                Prokudin-Gorskii), that contains the three inverted negatives 
                (top to bottom: blue, green, red)
    displacement_range -- range for the displacement points (default 10)
    '''
    
    i = imread(img_name)
    i.astype(uint8)
    normalize_image(i)

    # how much off the sides to crop off later...
    w = i.shape[1]
    w_5 = np.ceil(w * .05)
    
    # cut image into three peices, cropping out the borders    
    l = i.shape[0]
    b = i[w_5:(l/3) - w_5, w_5:-w_5]
    g = i[w_5 + (l/3):((l/3)*2) - w_5, w_5:-w_5]
    r = i[w_5 + ((l/3)*2):l-(l%3) - w_5, w_5:-w_5]
    
    displacements = get_displacement_vectors(displacement_range)
    
    g_match = best_match(func, b, g, displacements)
    r_match = best_match(func, b, r, displacements)
    
    result = zeros(b.shape + (3,), dtype=uint8)
    result[:,:,0] = shift(r, r_match)
    result[:,:,1] = shift(g, g_match)
    result[:,:,2] = b
    
    result = crop(result, max_displacement(r_match, g_match))
    
    return result

def sum_tup(tup1, tup2):
    return tuple(map(sum,zip(tup1,tup2)))
    
def mult_tup(t, s):
    return tuple([item * s for item in t])

def part2(img_name, func=ncc, displacement_range=20, pyramid_levels=5):
    ''' Return an image that has been combined to be in full colour format from 
    the three-channel input image that was given, with the help of an image
    pyramid.
    
    Keyword arguments:
    func -- corresponds to a function that implements some sort of patch-
            matching algorithm
    img_name -- name of the file with the 3-channelled photo (in the style of
                Prokudin-Gorskii), that contains the three inverted negatives 
                (top to bottom: blue, green, red)
    displacement_range -- range for the displacement points (default 10)
    '''
    
    i = imread(img_name)
    i.astype(uint8)
    normalize_image(i)
    
    pyramid = [i]
    for j in range(1, pyramid_levels):
        pyramid.insert(0, imresize(i, math.pow(0.5, j)))
        
    i = pyramid[0]
    i.astype(uint8)
    normalize_image(i)

    # how much off the sides to crop off later...
    w = i.shape[1]
    w_5 = np.ceil(w * .05)
    
    # cut image into three peices, cropping out the borders    
    l = i.shape[0]
    b = i[w_5:(l/3) - w_5, w_5:-w_5]
    g = i[w_5 + (l/3):((l/3)*2) - w_5, w_5:-w_5]
    r = i[w_5 + ((l/3)*2):l-(l%3) - w_5, w_5:-w_5]
    
    displacements = get_displacement_vectors(displacement_range)
    
    g_match = best_match(func, b, g, displacements)
    r_match = best_match(func, b, r, displacements)
    
    result = zeros(b.shape + (3,), dtype=uint8)
    result[:,:,0] = shift(r, r_match)
    result[:,:,1] = shift(g, g_match)
    result[:,:,2] = b
    result = crop(result, max_displacement(r_match, g_match))
    figure(); imshow(result)
    
    displacements_origin = get_displacement_vectors(max(displacement_range / 2, 5))
    for img in pyramid[1:]:

        # how much off the sides to crop off later...
        w_5 *= 2
        r_match  = mult_tup(r_match, 2)
        g_match = mult_tup(g_match, 2)
        
        displacements_r =  tuple([sum_tup(r_match, t) for t in displacements_origin])
        displacements_g =  tuple([sum_tup(g_match, t) for t in displacements_origin])
        print displacements_r, displacements_g
        # cut image into three peices, cropping out the borders    
        l = img.shape[0]
        b = img[w_5:(l/3) - w_5, w_5:-w_5]
        g = img[w_5 + (l/3):((l/3)*2) - w_5, w_5:-w_5]
        r = img[w_5 + ((l/3)*2):l-(l%3) - w_5, w_5:-w_5]
        
        r = shift(r, r_match)
        g = shift(g, g_match)
        
        # shift based on previous values
        r_match = best_match(func, b, r, displacements_r)
        g_match = best_match(func, b, r, displacements_g)

        r = shift(r, r_match)
        g = shift(g, g_match)
        
        result = zeros(b.shape + (3,), dtype=uint8)
        result[:,:,0] = r
        result[:,:,1] = g
        result[:,:,2] = b
    
        result = crop(result, max_displacement(r_match, g_match))
        figure(); imshow(result)
        
    return result


def ssd_ncc(func, img_name, displacement_range=10):
    ''' Show an image that displays the solution side by side with using
    SSD and NCC, respectively.
    
    Keyword arguments:
    func -- function that returns the result image (part1 or part2)
    img_name -- name of the file with the 3-channelled photo
    displacement_range -- range for the displacement points, default 10
    '''

    result_ssd = func(img_name, ssd,displacement_range)
    result_ncc = func(img_name, ncc, displacement_range)
    f = figure(figsize=(11, 6))
    f.add_subplot(1, 2, 1)
    imshow(result_ssd)
    f.add_subplot(1, 2, 2)
    imshow(result_ncc)


if __name__ == '__main__':
    
    plt.close("all")
    
#     if len(dir):
#         os.chdir(dir)
#     else:
    os.chdir('/Users/sangahhan/Workspace/School/CSC320/P1/images/')
    
#     # Part 1
#     
#     files = ['00757v.jpg', '00907v.jpg', '00911v.jpg','00106v.jpg',]
#     
#     for filename in files:
#         ssd_ncc(part1, filename)
#         
#     
    # Part 2

    os.chdir('/Users/sangahhan/Workspace/School/CSC320/P1/images/')
    files = []
    #imshow(part1('00822u.png', ncc))
    start_time = time.time()
    part2('00822u.png')
    print "--- %s seconds ---" % (time.time() - start_time) 
    