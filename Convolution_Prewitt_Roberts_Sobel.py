#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 06:23:52 2017

@author: julien
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:39 2016

@author: vmuser
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

I = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")[:,:,1]
#I = plt.imread("/home/vmuser/Downloads/classic_edge_detectors_1.0/lena.png")
#I = np.copy(RGB[:, :, 1])
#plt.imshow(I,cmap = plt.cm.gray)

R_row, R_col = I.shape

##########################

prewitt1 = np.array([[-0.333,-0.333,-0.333],[0,0,0],[0.333,0.333,0.333]])

prewitt2 = np.array([[-0.333,0,0.333],[-0.333,0,0.333],[-0.333,0,0.333]])

sobel1 = np.array([[-0.25,-0.5,-0.25],[0,0,0],[0.25,0.5,0.25]])

sobel2 = np.array([[-0.25,0,0.25],[-0.5,0,0.5],[-0.25,0,0.5]])

roberts1 = np.array([[-1,0],[0,1]])

roberts2 = np.array([[0,-1],[1,0]])

############################

#2D convolution with Roberts, Prewitt, Sobel

robertsG1 = signal.convolve2d(I, roberts1, mode='same', boundary='fill')

robertsG2 = signal.convolve2d(I, roberts2, mode='same', boundary='fill')

prewittG1 = signal.convolve2d(I, prewitt1, mode='same', boundary='fill')

prewittG2 = signal.convolve2d(I, prewitt2, mode='same', boundary='fill')

sobelG1 = signal.convolve2d(I, sobel1, mode='same', boundary='fill')

sobelG2 = signal.convolve2d(I, sobel2, mode='same', boundary='fill')

filters = [robertsG1,robertsG2,prewittG1,prewittG2,sobelG1,sobelG2]
filters_names = ['robertsG1','robertsG2','prewittG1','prewittG2','sobelG1','sobelG2']
############################
for i in range(0,6):
    plt.subplot(2,3,i+1)
    plt.imshow(filters[i],cmap='gray')
    plt.title(filters_names[i])
    plt.axis('off')
############################

prewittmaximum = np.sqrt(prewittG1**2 + prewittG2**2)

robertsmaximum = np.sqrt(robertsG1**2 + robertsG2**2)

sobelmaximum = np.sqrt(sobelG1**2 + sobelG2**2)

filters = [prewittmaximum,robertsmaximum,sobelmaximum]
filters_names = ['prewittmaximum','robertsmaximum','sobelmaximum']

############################
for i in range(0,3):
    plt.subplot(1,3,i+1)
    plt.imshow(filters[i],cmap='gray')
    plt.title(filters_names[i])
    plt.axis('off')
############################