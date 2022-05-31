#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:06:23 2022

@author: orochi
"""

import os
import numpy as np
import re
import shutil

def generate_training_split(num_batches, path=None):
    if path is None:
        path = './raw_data/RAL22_Paper/6_real_fall21_x5/'
    if type(num_batches) is not int:
        raise Exception('what the actual fuck dude, give me an integer')
        
    # load in the names of all the files we are working with
    grasp_successes = os.listdir(path + 'GRASP/new_pp5_labeled/test_set/successful')
    grasp_fails = os.listdir(path + 'GRASP/new_pp5_labeled/test_set/failed')
    grasp_successes.extend(os.listdir(path + 'GRASP/new_pp5_labeled/test_set/could_be_success'))
    
    
    # shuffle the successes and failes and split them into batches
    np.random.shuffle(grasp_successes)
    np.random.shuffle(grasp_fails)
    grasp_success_batches = np.array_split(grasp_successes, num_batches)
    grasp_fail_batches = np.array_split(grasp_fails, num_batches)


    # find which numbers we are using in the shuffled batches so that we get the same pick batches
    success_nums = []
    for thing in grasp_successes:
        temp = re.search('\d+', thing)
        success_nums.append(temp.group(0))
    success_batches = np.array_split(success_nums, num_batches)
    fail_nums = []
    for thing in grasp_fails:
        temp = re.search('\d+', thing)
        fail_nums.append(temp.group(0))
    fail_batches = np.array_split(fail_nums, num_batches)


    # divide the pick titles into batches matching the grasp batches


    # Set up the file tree in the way we expect it in csv_process
    os.makedirs(path + 'training_testing_split/set1/GRASP/test_set/failed/')
    os.makedirs(path + 'training_testing_split/set1/GRASP/test_set/successful/')
    os.makedirs(path + 'training_testing_split/set1/GRASP/test_set/could_be_success/')
    shutil.copytree(path + 'training_testing_split/set1/GRASP/test_set/', path + 'training_testing_split/set1/GRASP/traning_set/')
    shutil.copytree(path + 'training_testing_split/set1/GRASP/', path + 'training_testing_split/set1/PICK/')
    for i in range(num_batches-1):
        shutil.copytree(path + 'training_testing_split/set1', path + 'training_testing_split/set' + str(i+2)+'/')
    
    # copy the files listed in the batches to their apropriate file path
    
generate_training_split(4)