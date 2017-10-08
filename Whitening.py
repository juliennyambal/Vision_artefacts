# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:39 2016

@author: julien
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

#I = plt.imread("/home/julien_nyambal/Desktop/computervision3/kodak/IMAGE-0.bmp")[:,:,1]
I = plt.imread("/home/julien/Documents/Python_tests/Vision_filters/000456.jpg")[:,:,1]
plt.imshow(I,cmap = plt.cm.gray)
plt.imshow(I)
#plt.show()
R_row, R_col = I.shape
I_result = np.copy(I)
mean = 0
numeratorM = 0
numeratorS = 0
variance =0
f = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
#plt.imshow(f,cmap = plt.cm.gray)
#plt.show()

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):
		numeratorM = (numeratorM + I[i][j])
mean = numeratorM/ (R_row*R_col)

#print "mean ", mean

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):		
		normalized_pixel = (I[i][j]- mean)**2
		numeratorS = numeratorS + normalized_pixel
		
std = numeratorS/ (R_row*R_col)

#print "std ", std

for i in np.arange(0, R_row):
    for j in np.arange(0, R_col):		
		normalized_pixel = I[i][j]- mean
		variance = std**(0.5)
		I_result[i][j] = normalized_pixel/variance
		
plt.figure()
plt.subplot(121)
plt.imshow(I, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Input")

plt.subplot(122)
plt.imshow(I_result, cmap = plt.cm.gray)
plt.axis("off")
plt.title("Whitening")

plt.show()





