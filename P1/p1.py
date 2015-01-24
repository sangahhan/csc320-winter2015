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
    step -- amount of space between points
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
            result = np.lib.pad(img[:-M,:-N],((M, 0),(N, 0)),mode="constant")
        else:
            result = np.lib.pad(img[:-M,N:],((M, 0),(0, N)),mode="constant")
    else:
        if n > 0:
            result = np.lib.pad(img[M:,:-N],((0, M),(N, 0)),mode="constant")
        else:
            result = np.lib.pad(img[M:,N:],((0, M),(0, N)),mode="constant")
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
    img_name -- name of the file with the 3-channelled photo (in the style of
                Prokudin-Gorskii), that contains the three inverted negatives 
                (top to bottom: blue, green, red)
    func -- corresponds to a function that implements some sort of patch-
            matching algorithm
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
    
    # construct final image
    result = zeros(b.shape + (3,), dtype=uint8)
    result[:,:,0] = shift(r, r_match)
    result[:,:,1] = shift(g, g_match)
    result[:,:,2] = b
    result = crop(result, max_displacement(r_match, g_match))
    
    return result


def sum_tup(tup1, tup2):
    ''' Given two tuples, return a tuple that contains the sum of their 
    corresponding indices.
    '''
    
    return tuple(map(sum,zip(tup1,tup2)))
    
    
def mult_tup(t, s):
    ''' Given a tuple and some scalar (number), return a tuple that contains all 
    the items from the original multiplied by that number.
    
    Keyword arguments:
    t -- the original tuple
    s -- number to multiply the items by
    '''
    
    return tuple([item * s for item in t])


def part2(img_name, displacement_range=10, func=ncc, min_percent=0.03):
    ''' Return an image that has been combined to be in full colour format from 
    the three-channel input image that was given, with the help of an image
    pyramid.
    
    Keyword arguments:
    img_name -- name of the file with the 3-channelled photo (in the style of
                Prokudin-Gorskii), that contains the three inverted negatives 
                (top to bottom: blue, green, red)
    displacement_range -- range for the displacement points (default 10)
    func -- corresponds to a function that implements some sort of patch-
            matching algorithm (default ncc)
    min_percent -- a fraction that corresponds to the percent of the size of the
                    image in the highest level of the pyramid wrt the original
                    (default 0.03)
    '''
    
    i = imread(img_name)
    i.astype(uint8)
    normalize_image(i)
    
    pyramid_levels = max(int(math.floor(math.log(min_percent, 0.5))), 5)
    
    # construct the pyramid via a list
    pyramid = [i]
    for j in range(1, pyramid_levels):
        pyramid.insert(0, imresize(i, math.pow(0.5, j)))

    # start @ the smallest
    i = pyramid[0]
    i.astype(uint8)

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

    for img in pyramid[1:]:
        
        # decrement the range of displacements to check
        displacement_range = max(displacement_range-1, 3)
        displacements = get_displacement_vectors(displacement_range)
        
        # how much off the sides to crop off later...
        w_5 *= 2
        r_match  = mult_tup(r_match, 2)
        g_match = mult_tup(g_match, 2)
        
        # add the displacements to the match from previous pyramid level
        displacements_r =  tuple([sum_tup(r_match, t) for t in displacements])
        displacements_g =  tuple([sum_tup(g_match, t) for t in displacements])
        
        # cut image into three peices, cropping out the borders    
        l = img.shape[0]
        b = img[w_5:(l/3) - w_5, w_5:-w_5]
        g = img[w_5 + (l/3):((l/3)*2) - w_5, w_5:-w_5]
        r = img[w_5 + ((l/3)*2):l-(l%3) - w_5, w_5:-w_5]
        
        R = shift(r, r_match)
        r_match_displaced = best_match(func, b, R, displacements)
        G = shift(g, g_match)
        g_match_displaced = best_match(func, b, R, displacements)
        
        r_match  = sum_tup(r_match, r_match_displaced)
        g_match = sum_tup(g_match, g_match_displaced)

    # at this point, the displacements in r_match and g_match should be okay
    r = shift(r, r_match)
    g = shift(g, g_match)
    
    # construct final image
    result = zeros(b.shape + (3,), dtype=uint8)
    result[:,:,0] = r
    result[:,:,1] = g
    result[:,:,2] = b
    result = crop(result, max_displacement(r_match, g_match))
    
    return result


def ssd_ncc(func, img_name, displacement_range=10):
    ''' Show an image that displays the solution for the image with the given 
    name side by side, using SSD and NCC, respectively.
    
    Keyword arguments:
    func -- function that returns the result image (part1 or part2)
    img_name -- name of the file with the 3-channelled photo
    displacement_range -- range for the displacement points (default 10)
    '''

    result_ssd = func(img_name, ssd,displacement_range)
    result_ncc = func(img_name, ncc, displacement_range)
    f = figure(figsize=(11, 6))
    f.add_subplot(1, 2, 1)
    imshow(result_ssd)
    f.add_subplot(1, 2, 2)
    imshow(result_ncc)


def print_time(func, args=[], show_image=True, unit="minutes"): 
    ''' Print the time that the given function takes to produce a solution.
    
    Keyword arguments:
    func -- function that returns the result image (part1 or part2)
    args -- the arguments to pass into the function in a list
    show_image -- a boolean that determines displaying the solution image
    unit -- time units to display; seconds|minutes|hours
    '''
    
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    if show_image:
        figure(); imshow(result)
    
    unit = unit.lower()
    convert_seconds = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600
    }
    print "> %s(%s): %s %s" % (func.__name__,
                            ", ".join([str(arg) for arg in args]), 
                            (end_time - start_time) / convert_seconds[unit], 
                            unit)


def part1_test():
    ''' Display comparisons between using SSD and NCC to align lower resolution 
    images. Also, display an example of using the non-zero mean NCC function. 
    Output order should correspond to the order in the report (p1.pdf).
    '''
    files = ["00757v.jpg", "00907v.jpg", "00911v.jpg","00106v.jpg"]
    
    for filename in files:
        ssd_ncc(part1, filename)
        if filename is "00911v.jpg":
            ssd_ncc(part1, filename, 15)


def part2_test(show_image=True):
    ''' Obtain the time to run NCC on a smaller image using the function for
    part 1. Than display examples of combining NCC with the use of an image 
    pyramid to align higher-resolution images. Print the runtimes to compute the
    solution for each image. Output order should correspond to the order in the 
    report (p1.pdf).
    '''
    
    # runtime from part 1
    print_time(part1, ['00106v.jpg', ncc], False, "seconds")
    
    # running part 2
    files = ["00128u.png", "01047u.png", "00458u.png", "00822u.png"]
    
    for filename in files:
        print_time(part2, [filename], show_image)


def run_tests(part_num=None):
    ''' Run tests for each part in the project, depending on the given part
    number. Trying to run part 0 will exit the program, as there is no code to 
    test for part 0. Any input besides 1 or 2 (corresponding with part 1 & part
    2) will simply run all tests. 
    '''
    
    plt.close("all")
    if part_num is 1:
        print "> Running test for part 1."
        part1_test()
        print "> Completed test for part 1."
    elif part_num is 2:
        print "> Running test for part 2."
        part2_test()
        print "> Completed test for part 2."
    elif part_num is 0:
        print "> Goodbye."
    else:
        print "> Running test for all parts."
        print "> Running test for part 1."
        part1_test()
        print "> Completed test for part 1."
        print "> Running test for part 2."
        part2_test()
        print "> Completed test for part 2."
    

if __name__ == "__main__":
    
    #os.chdir("/Users/sangahhan/Workspace/School/CSC320/P1/images/")
    
    print ("> %s" + "\n> %s" * 3) % ("g2sangah, CSC320H1 Winter 2015", 
                    "Project 1: The Prokudin-Gorskii Colour Photo Collection",
                    "Input which part of the project you want to run.",
                    "Invalid input (e.g. empty input) will run all parts.")
    num = raw_input("> Please type, 0, 1 or 2, then hit [ENTER]: ")
    plt.ion()
    try:
        num_int = int(num)
        run_tests(num_int)
    except ValueError:
        run_tests()
    