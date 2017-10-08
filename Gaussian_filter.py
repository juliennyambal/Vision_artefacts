#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 05:37:46 2017

@author: julien
"""
import numpy as np
import matplotlib.pyplot as plt

#Gaussian filter

#G(x,y) = 1/(2*pi.sigma^2)) * e^-((x^2+y^2)/2*sigma^2)

sigma = 2

x = np.linspace(-10,10,100)[:,np.newaxis] # or reshape(100,1)
y = np.linspace(-10,10,100)[np.newaxis,:] # or reshape(1, 100)

#2*pi.sigma^2
denomimator_out = 2 * np.pi * sigma**2

#1/(2*pi.sigma^2))
outside = 1 / denomimator_out

#2*sigma^2
denominator_in = 2 * sigma**2

#x^2+y^2
nominator_in = -(x**2 + y**2)

#((x^2+y^2)/2*sigma^2)
in_exp = nominator_in / denominator_in

exp_sum = np.exp(in_exp)


gaussian_filter = outside * exp_sum

plt.imshow(gaussian_filter)
plt.colorbar()