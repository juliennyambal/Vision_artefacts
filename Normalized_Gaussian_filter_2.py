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
I = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")[:,:,1]
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

Ix = signal.convolve2d(I, Hx, mode='same')
Iy = signal.convolve2d(I, Hy, mode='same')

G = np.sqrt(Ix**2 + Iy**2)
plt.figure()
plt.subplot(221)
plt.imshow(I, cmap = plt.cm.gray)
plt.title("$I$")
plt.axis('off')
plt.subplot(222)
plt.imshow(Ix, cmap = plt.cm.gray)
plt.title("$Ix$")
plt.axis('off')
plt.subplot(223)
plt.imshow(Iy, cmap = plt.cm.gray)
plt.title("$Iy$")
plt.axis('off')
plt.subplot(224)
plt.imshow(G, cmap = plt.cm.gray)
plt.title("$G$")
plt.axis('off')
plt.show()