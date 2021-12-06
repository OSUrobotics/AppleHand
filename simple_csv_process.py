# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:14:34 2021

@author: nigel
"""

import copy
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import interp1d
from scipy.signal import resample
import math
import bisect
import pickle as pkl
import csv
from raw_csv_process import unpack_arr

def simple_process_data(path):
    """
    Reads apple picking data saved in the given path, removes datapoints
    that are not recorded on all channels, removes dead time in the pick,
    downsamples high frequency data, generates labels for all picks and saves
    processed data as npy files for later use.
    @param path - Filepath containing rosbags and csvs from apple picking"""
    success_keys = ['s', 'y', 'yes', 'success']
    fail_keys = ['f', 'n', 'no', 'fail']
    folder_names = ['successful_picks', 'failed_picks']
    pcount = 0
    ncount = 0
    episode_times = [[-1, -1]]
    states = []
    labels = []
    lens = []
    # This loop iterates through all the folders in the path folder, which contains different folders for each set of starting noises
    for folder in folder_names:
        csv_files = os.listdir(path + '/' + folder)

        # This loop iterates through the folders in the selected folder in path. These folders contain csvs with all the
        # episode data and have the same pick number as the csv label file in the folder above
        for datafile in csv_files:
            indicies = [len(states), -1]
            print('opening up name', datafile)
            with open(path + '/' + folder + '/' + datafile, 'r') as csv_file:
                reader = csv.reader(csv_file)
                temp = False
                for row in reader:
                    if temp:
                        states.append(row[1:])
                    else:
                        temp = True
            indicies[1] = len(states)
            if folder == 'successful_picks':
                labels.extend([[1,1,1,1,1,1,1]] * (indicies[1] - indicies[0]))
                pcount += 1
            else:
                labels.extend([[0,0,0,0,0,0,0]] * (indicies[1] - indicies[0]))
                ncount += 1
            episode_times.append(indicies.copy())

    # this removes the first instance of episode times which is just a placeholder
    episode_times.pop(0)
    episode_times = np.array(episode_times)
    np.random.shuffle(episode_times)
    test_inds = []
    train_inds = []
    # print(episode_times[:, -1])
    # for part in range(np.max(episode_times[:, -1])):
    #     limited_times = episode_times[episode_times[:, -1] == part]
    # np.random.shuffle(limited_times)  # episode_times)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.2 * len(episode_times))
    # print(episode_times)
    train_inds.extend([episode_times[i][0], episode_times[i][1]] for i in range(len(episode_times) - test_portion))
    test_inds.extend([episode_times[-i - 1][0], episode_times[-i - 1][1]] for i in range(test_portion))

    # this separates the episode into train and test episodes
    train_state = [states[train_inds[i][0]:train_inds[i][1]] for i in range(len(train_inds))]
    train_label = [labels[train_inds[i][0]:train_inds[i][1]] for i in range(len(train_inds))]
    test_state = [states[test_inds[i][0]:test_inds[i][1]] for i in range(len(test_inds))]
    test_label = [labels[test_inds[i][0]:test_inds[i][1]] for i in range(len(test_inds))]
    # print(train_label, test_label)

    train_label = np.array(unpack_arr(train_label))
    test_label = np.array(unpack_arr(test_label))
    test_trim_inds = np.ones(len(test_label))
    test_state = np.array(unpack_arr(test_state))
    train_state = np.array(unpack_arr(train_state))
    train_trim_inds = np.ones(len(train_label))
    # print(test_label)
    train_label = train_label.astype(float)
    test_label = test_label.astype(float)
    test_trim_inds = test_trim_inds.astype(float)
    test_state = test_state.astype(float)
    train_state = train_state.astype(float)
    train_trim_inds = train_trim_inds.astype(float)
    print('average from train and test labels', np.mean(train_label), np.mean(test_label))
    print('lens of labels', np.shape(train_label), np.shape(test_label))
    data_file = {'train_state': train_state, 'train_label': train_label,
                 'test_state': test_state, 'test_label': test_label,
                 'train_reduce_inds': train_trim_inds, 'test_reduce_inds': test_trim_inds,
                 'train_indexes': train_inds, 'test_indexes': test_inds}
    file = open('apple_dataset.pkl', 'wb')
    pkl.dump(data_file, file)
    file.close()
    print('all files saved')
