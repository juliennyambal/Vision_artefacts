#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 06:20:43 2017

@author: julien
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

dt = 0.1
x= np.arange(-3.0,3.0+dt , dt)[:, np.newaxis]
y= np.arange(-3.0,3.0+dt , dt)[np.newaxis,: ]
std = 0.1
F = np.exp(-(x**2 + y**2) / (2.0 * std))
plt.figure()
plt.imshow(F, cmap = plt.cm.gray)
plt.show()