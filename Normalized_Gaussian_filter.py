#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 06:55:51 2017

@author: julien
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:39 2016

@author: Julien Nyambal
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

#I = plt.imread("/home/vmuser/Downloads/kodak/IMAGE-0.bmp")[:,:,1]
I = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")
#plt.imshow(I,cmap = plt.cm.gray)

#I = I * 255

variance = 1

ns = 21

I.astype(np.uint8)
#plt.imshow(I, cmap = plt.cm.gray)
x = np.linspace(-3*np.sqrt(variance),3*np.sqrt(variance), ns)[np.newaxis, :]
y = np.linspace(-3*np.sqrt(variance),3*np.sqrt(variance), ns)[:, np.newaxis]

H = np.exp(- (x**2 + y**2)/(2 * variance))

Hx = -x / variance * H

Hy = -y / variance * H

#normalization of the values
Hx = Hx - np.average(Hx)
Hx = Hx / np.sum(np.abs(Hx))
Hy = Hy - np.average(Hy)
Hy = Hy / np.sum(np.abs(Hy))
#normalization of the values

plt.figure()
plt.subplot(131)
plt.imshow(H, cmap = plt.cm.gray)
plt.title("$H$")
plt.subplot(132)
plt.imshow(Hx, cmap = plt.cm.gray)
plt.title("$H_{x}$")
plt.subplot(133)
plt.imshow(Hy, cmap = plt.cm.gray)
plt.title("$H_{y}$")
plt.show()