#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:09:14 2022

@author: orochi
"""

import matplotlib.pyplot as plt
import numpy as np

x = range(40)
y1 = []
y2 = []
for i in range(5):
    y1.append(0.5)
    y2.append(0.5)

for i in range(10):
    y1.append(0.5+0.5*(i+1)/10)
    y2.append(0.5-0.5*(i+1)/10)

for i in range(25):
    y1.append(1)
    y2.append(0)
    
plt.plot(x,y1)
plt.plot(x,y2)
#plt.legend(['successful pick','failed pick'])
plt.vlines(20,-1,2,color='r')
plt.xlabel('timestep')
plt.ylabel('LSTM Label')
plt.ylim([-0.1,1.1])
plt.show()