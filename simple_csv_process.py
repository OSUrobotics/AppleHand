# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:14:34 2021

@author: Nigel Swenson
"""

import os
import numpy as np
import pickle as pkl
import csv
from utils import unpack_arr


def process_data_iterable(path):
    """
    Reads apple picking data saved in the given path, removes datapoints
    that are not recorded on all channels, removes dead time in the pick,
    downsamples high frequency data, generates labels for all picks and saves
    processed data as npy files for later use.
    @param path - Filepath containing rosbags and csvs from apple picking"""
    folder_names = ['successful_picks', 'failed_picks']
    big_names = ['training_set', 'test_set']
    pcount = 0
    ncount = 0
    states = {'training_set': [], 'test_set': []}
    labels = {'training_set': [], 'test_set': []}

    # This loop iterates through all the folders in the path folder, which contains different folders for each set of starting noises
    for top_folder in big_names:
        for folder in folder_names:
            csv_files = os.listdir(path + '/' + top_folder + '/' + folder)
            # This loop iterates through the folders in the selected folder in path. These folders contain csvs with all the
            # episode data and have the same pick number as the csv label file in the folder above
            for datafile in csv_files:
                episode_state = []
                print('opening up name', datafile)
                with open(path + '/' + top_folder + '/' + folder + '/' + datafile, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    temp = False
                    for row in reader:
                        if temp:
                            episode_state.append(row[1:])
                        else:
                            temp = True
                if len(episode_state) > 0:
                    episode_state = np.array(episode_state)
                    episode_state = episode_state.astype(float)
                    if folder == 'successful_picks':
                        labels[top_folder].append([1] * len(episode_state))
                        pcount += 1
                    else:
                        labels[top_folder].append([0] * len(episode_state))
                        ncount += 1
                        print(top_folder)
                    states[top_folder].append(episode_state.copy())

    data_file = {'train_state': states['training_set'], 'train_label': labels['training_set'],
                 'test_state': states['test_set'], 'test_label': labels['test_set']}
    file = open('apple_dataset.pkl', 'wb')
    pkl.dump(data_file, file)
    file.close()
    print('all files saved')


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
    big_names = ['training_set', 'test_set']
    pcount = 0
    ncount = 0
    episode_times = {'training_set':[[-1, -1]], 'test_set':[[-1, -1]]}
    states = {'training_set': [], 'test_set': []}
    labels = {'training_set': [], 'test_set': []}
    lens = []
    # This loop iterates through all the folders in the path folder, which contains different folders for each set of starting noises
    for top_folder in big_names:
        for folder in folder_names:
            csv_files = os.listdir(path + '/' + top_folder + '/' + folder)
            # This loop iterates through the folders in the selected folder in path. These folders contain csvs with all the
            # episode data and have the same pick number as the csv label file in the folder above
            for datafile in csv_files:
                indicies = [len(states[top_folder]), -1]
                print('opening up name', datafile)
                with open(path + '/' + top_folder + '/' + folder + '/' + datafile, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    temp = False
                    for row in reader:
                        if temp:
                            states[top_folder].append(row[1:])
                        else:
                            temp = True
                indicies[1] = len(states[top_folder])
                if folder == 'successful_picks':
                    labels[top_folder].extend([[1, 1, 1, 1, 1, 1, 1]] * (indicies[1] - indicies[0]))
                    pcount += 1
                else:
                    labels[top_folder].extend([[0, 0, 0, 0, 0, 0, 0]] * (indicies[1] - indicies[0]))
                    ncount += 1
                episode_times[top_folder].append(indicies.copy())

    # this removes the first instance of episode times which is just a placeholder
    episode_times['training_set'].pop(0)
    episode_times['test_set'].pop(0)
    for key in episode_times.keys():
        episode_times[key].pop(0)
        episode_times[key] = np.array(episode_times[key])
        np.random.shuffle(episode_times[key])
    test_inds = []
    train_inds = []

    trim_inds = {}
    # this separates the episode into train and test episodes
    for key in states.keys():
        states[key] = [states[key][episode_times[key][i][0]:episode_times[key][i][1]] for i in range(len(episode_times[key]))]
        labels[key] = [labels[key][episode_times[key][i][0]:episode_times[key][i][1]] for i in range(len(episode_times[key]))]
        # print(train_label, test_label)

        labels[key] = np.array(unpack_arr(labels[key]))
        states[key] = np.array(unpack_arr(states[key]))
        trim_inds[key] = np.ones(len(labels[key]))
        # print(test_label)
        labels[key] = labels[key].astype(float)
        states[key] = states[key].astype(float)
        trim_inds[key] = trim_inds[key].astype(float)

    print('average from train and test labels', np.mean(labels['training_set']), np.mean(labels['test_set']))
    print('lens of labels', np.shape(labels['training_set']), np.shape(labels['test_set']))
    data_file = {'train_state': states['training_set'], 'train_label': labels['training_set'],
                 'test_state': states['test_set'], 'test_label': labels['test_set'],
                 'train_reduce_inds': trim_inds['training_set'], 'test_reduce_inds': trim_inds['test_set'],
                 'train_indexes': episode_times['training_set'], 'test_indexes': episode_times['test_set']}
    file = open('apple_dataset.pkl', 'wb')
    pkl.dump(data_file, file)
    file.close()
    print('all files saved')

