#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 07:05:12 2017

@author: julien
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

I = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")[:,:,0]
plt.imshow(I,cmap = plt.cm.gray)
plt.imshow(I)
#plt.show()
R_row, R_col = I.shape
print("R_row",R_row)
print("R_col",R_col)

prewitt1 = np.array([[-0.333,-0.333,-0.333],[0,0,0],[0.333,0.333,0.333]])

prewitt2 = np.array([[-0.333,0,0.333],[-0.333,0,0.333],[-0.333,0,0.333]])

prewittG1 = signal.convolve2d(I, prewitt1, mode='same', boundary='fill')

prewittG2 = signal.convolve2d(I, prewitt2, mode='same', boundary='fill')

prewittmaximum = np.sqrt(prewittG1**2 + prewittG2**2)

maxP = 0

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if prewittmaximum[i][j] >= maxP:
            maxP = prewittmaximum[i][j]

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
        if prewittmaximum[i][j] >= 0.1 * maxP:
            prewittmaximum[i][j] = 1
        else:    
            prewittmaximum[i][j] = 0
            

plt.figure()
plt.subplot(121)
plt.imshow(I, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Input")

plt.subplot(122)
plt.imshow(prewittmaximum,cmap = plt.cm.gray)
plt.axis("off")
plt.title("Prewitt")

plt.show()