#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:53:55 2022

@author: orochi
"""

import os
import numpy as np
import pickle as pkl
import csv
from utils import unpack_arr
from copy import deepcopy
import re
import matplotlib.pyplot as plt


def process_folder(root_path,paths):
    AUC_dict = {}
    loss_dict = {}
    for path in paths:
        with open(root_path+path,'rb') as file:
            temp_dict = pkl.load(file)
        size = re.search('input_size=\d+',temp_dict['ID'])
        try:
            AUC_dict[size.group(0)].append(temp_dict['AUC'])
            loss_dict[size.group(0)].append(temp_dict['loss'])
        except KeyError:
            AUC_dict[size.group(0)] = [temp_dict['AUC']]
            loss_dict[size.group(0)] = [temp_dict['loss']]
    return AUC_dict, loss_dict

root_dir = './generated_data/'
pick_dir = 'pick_ablation_data/'
grasp_dir = 'grasp_ablation_data/'
full_dir = 'full_ablation_data/'

grasps = os.listdir(root_dir+grasp_dir)
#picks = os.listdir(root_dir+pick_dir)
fulls = os.listdir(root_dir+full_dir)

temp =[]
for name in fulls:
    if name not in grasps:
        temp.append(name)

fulls = temp
grasp_AUC, grasp_loss = process_folder(root_dir+grasp_dir,grasps)
full_AUC, full_loss = process_folder(root_dir+full_dir,fulls)

with open(root_dir+'grasp_epochs=50_hidden_nodes=10_input_size=33_hidden_layers=306_02_22_1220.pkl', 'rb') as file:
    comparison_dict = pkl.load(file)

#for data in full_AUC['input_size=33']:
#    plt.plot(range(len(data)),data)
for data in full_loss['input_size=3']:
    plt.plot(range(len(data)-1),data[1:])
plt.show()
plt.plot(range(len(comparison_dict['loss'])-1), comparison_dict['loss'][1:])
plt.show()
plt.legend(['ablation_1', 'ablation_2', 'ablation_3', 'expected_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')


