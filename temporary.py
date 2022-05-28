#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:14:50 2022

@author: orochi
"""
import numpy as np
import matplotlib.pyplot as plt
a  = [[0.95,	91,	0.85,	85],[0.978,	95,	0.814,	80],[0.95,	92.3,	0.833,	78.6]]
b  = [[0.938,	90,	0.764,	82.9],[0.967,	90.5,	0.895,	88.6],[0.911,	87.9,	0.773,	77.1]]
c = [[0.986,	94.5,	0.909,	88.6],[0.988,	96.1,	0.865,	82.9],[0.989,	95.8,	0.795,	81.4]]
a = np.array(a)
b = np.array(b)
c = np.array(c)

amean = np.average(a,axis=0)
bmean = np.average(b,axis=0)
cmean = np.average(c,axis=0)
astd = np.std(a,axis=0)
bstd = np.std(b,axis=0)
cstd = np.std(c,axis=0)


print(astd)
nums = [0.0001, 0.00005, 0.0005]

#plt.errorbar(nums,[amean[2],bmean[2],cmean[2]],)
plt.boxplot(a)
plt.boxplot(b)
plt.boxplot(c)