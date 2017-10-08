#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 07:08:31 2017

@author: julien
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

I = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")[:,:,0]
#plt.imshow(I,cmap = plt.cm.gray)
#plt.imshow(I)
#plt.show()

filter1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
filter2 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
print filter1, filter2
R_row, R_col = I.shape

I_filter1 = signal.convolve2d(I, filter1, mode='same', boundary='fill')
I_filter2 = signal.convolve2d(I, filter2, mode='same', boundary='fill')


plt.figure()
plt.subplot(121)
plt.imshow(I_filter1,cmap = plt.cm.gray)
plt.axis("off")
plt.title("$I_{Filter1}$")

plt.subplot(122)
plt.imshow(I_filter2,cmap = plt.cm.gray)
plt.axis("off")
plt.title("IFilter2")

plt.show()