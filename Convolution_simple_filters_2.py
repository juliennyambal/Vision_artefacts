#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 07:11:35 2017

@author: julien
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

RGB = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")
I = np.copy(RGB[:, :, 0])
#plt.imshow(I,cmap = plt.cm.gray)

F = np.array([[-1,0,1]])
F1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
F2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
F3 = np.array([[0,0,0],[1,-2,1],[0,0,0]])
F4 = np.array([[0,1,0],[0,-2,0],[0,1,0]])


V = signal.convolve2d(I, F, mode='same', boundary='fill')
V1 = signal.convolve2d(I, F1, mode='same', boundary='fill')
V2 = signal.convolve2d(I, F2, mode='same', boundary='fill')
V3 = signal.convolve2d(I, F3, mode='same', boundary='fill')
V4 = signal.convolve2d(I, F4, mode='same', boundary='fill')

plt.figure()

plt.subplot(221)
plt.imshow(V, cmap = plt.cm.gray)
plt.axis('off')

plt.subplot(222)
plt.imshow(V1, cmap = plt.cm.gray)
plt.axis('off')

plt.subplot(223)
plt.imshow(V2, cmap = plt.cm.gray)
plt.axis('off')

plt.subplot(224)
plt.imshow(V4, cmap = plt.cm.gray)
plt.axis('off')

plt.show()