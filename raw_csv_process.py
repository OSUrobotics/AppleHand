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


def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    @param long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr


def downsample(signal, times):
    """
    Takes a timestamped signal or set of signals and downsamples to fit a
    desired set of times. Better than scipy resample because it works for
    arbitrary times instead of regular times
    @param signal - Pandas dataframe containing a time param and some signal
    @param times - list or array with times to split the signal"""
    start = time.time()
    time_series = list(signal['Time'])
    split_points = [bisect.bisect_right(time_series, i) for i in times]
    split_signal = np.array_split(signal.drop('Time', axis=1), split_points)
    split_signal.pop(0)
    for i in range(len(split_signal)):
        split_signal[i] = split_signal[i].mean(axis='index')
    split_signal = pd.concat(split_signal, axis=1)
    split_signal = split_signal.transpose()
    for i in range(len(split_signal)):
        print(split_signal[i])
    return split_signal


def downsample_regular(signal, num_points):
    """
    Takes a timestamped signal or set of signals and downsamples to fit a
    desired number of points. This assumes that the points in signal are
    sampled at a fixed frequency. Much faster than downsample above
    @param signal - Pandas dataframe containing a time param and some signal
    @param num_points - int with number of poitns in the downsampled signal"""
    new_signal = signal.drop('Time', axis=1)
    new_signal = resample(new_signal, num_points)
    return new_signal


def process_data(path):
    """
    Reads apple picking data saved in the given path, removes datapoints 
    that are not recorded on all channels, removes dead time in the pick, 
    downsamples high frequency data, generates labels for all picks and saves
    processed data as npy files for later use.
    @param path - Filepath containing rosbags and csvs from apple picking"""
    success_keys = ['s', 'y', 'yes', 'success']
    fail_keys = ['f', 'n', 'no', 'fail']
    big_names = os.listdir(path)
    states = []
    labels = []
    csv_name_list = ['applehand-finger1-imu.csv', 'applehand-finger1-jointstate.csv',
                     'applehand-finger2-imu.csv', 'applehand-finger2-jointstate.csv',
                     'applehand-finger3-imu.csv', 'applehand-finger3-jointstate.csv',
                     'oint_states.csv', 'rench.csv']
    use_name = 'apple_trial_events.csv'
    sub_label = ['f1_imu', 'f1_joint', 'f2_imu', 'f2_joint', 'f3_imu', 'f3_joint',
                 'joint_states', 'wrench_force']
    count = 0
    episode_times = [[-1, -1]]
    folder_num = 0

    # This loop iterates through all the folders in the path folder, which contains different folders for each set of starting noises
    for big_name in big_names:
        temp = os.listdir(path + big_name)
        indicies = []

        # This makes sure that we only look at the subfolders rather than csv and bag files
        for i in range(len(temp)):
            if ('.csv' in temp[i]) | ('.bag' in temp[i]):
                pass
            else:
                indicies.append(i)
        temp = np.array(temp)
        names = temp[indicies]
        print('Starting overall folder', big_name)

        # This loop iterates through the folders in the selected folder in path. These folders contain csvs with all the
        # episode data and have the same pick number as the csv label file in the folder above
        for name in names:
            print('opening up name', name)
            states = []
            labels = []
            lens = []
            latest_start = 0
            earliest_end = math.inf
            label_name_key_phrase = name[0:8]

            # this reads through all the csvs that match the format we expect and extracts the data we care about
            try:
                for i in range(len(csv_name_list)):
                    data = pd.read_csv(path + big_name + '/' + name + '/' + csv_name_list[i])
                    # This ignores the header and name columns while renaming columns that appear in multiple files
                    for column_name in data.columns:
                        if ('header' in column_name) | ('name' in column_name):
                            data = data.drop(column_name, axis=1)
                        elif ('acceleration' in column_name) | ('velocity' in column_name) | (
                                'position' in column_name) | ('effort' in column_name):
                            data = data.rename(columns={column_name: sub_label[i] + column_name})
                    states.append(data)
                    lens.append(len(data))
                    if latest_start < data['Time'][0]:
                        latest_start = data['Time'][0]
                    thing = np.array(data['Time'])[-1]
                    if earliest_end > thing:
                        earliest_end = thing

                # This looks for a csv file outside of the subfolder with the same pick number. If there is one of these it
                # corresponds to the same attempt. If there are 0 or 2 we don't have a match so we ignore this trial
                dupe_flag = False
                for i in range(len(temp)):
                    # print(label_name_key_phrase,temp[i][0:8], (label_name_key_phrase == temp[i][0:8]) & ('.csv' in temp[i]))
                    if (label_name_key_phrase == temp[i][0:8]) & ('.csv' in temp[i]):
                        if dupe_flag:
                            print('multiple csvs with the same number found, disregarding this file')
                            count += 1
                            raise FileNotFoundError
                        label_file = temp[i]
                        dupe_flag = True
                with open(path + big_name + '/' + label_file, 'r') as csv_file:
                    reader = csv.reader(csv_file)
                    for row in reader:
                        pass
                label_data = row

                # This reads the event file to determine when the gripper closed and when we started pulling
                use_data = pd.read_csv(path + big_name + '/' + name + '/' + use_name)
            except FileNotFoundError:
                print(f'missing file in {big_name} {name}. skipping to next folder')
                count += 1
                continue
            except NameError:
                print(f'NAME ERROR missing file in {big_name} {name}. skipping to next folder')
                count += 1
                continue

            # Now that we have all the states saved in states, the labels saved in label_data, the start and end times saved
            # in latest_start and earliest end, and the event times in use data, we need to crop out the state data that
            # happens either before all the topics are publishing or after some of them have stopped and interpolate the
            # data so that all the topics appear at 500 Hz
            min_len_index = np.argmin(lens)
            time_for_downsample = np.array(states[min_len_index]['Time'])
            time_for_downsample = time_for_downsample[
                (time_for_downsample > latest_start) & (time_for_downsample < earliest_end)]
            episode_states = np.array([])
            #            num_samples = len(time_for_downsample)
            use_part = np.zeros([len(time_for_downsample)])
            use_part[(time_for_downsample > use_data['Time'][0]) & (time_for_downsample < use_data['Time'][0] + 2)] = 1
            use_part[time_for_downsample > use_data['Time'][1]] = 1
            num_points = len(time_for_downsample)
            for state in states:
                trimmed_state = state[(state['Time'] > latest_start) & (state['Time'] < earliest_end)]
                #                downsampled_state = downsample(trimmed_state,time_for_downsample)
                #                interpolateinator = interp1d(state['Time'], state.drop('Time', axis = 1), axis = 0, kind = 'previous')
                if len(trimmed_state) > len(time_for_downsample):
                    try:
                        episode_states = np.append(episode_states, downsample_regular(trimmed_state, num_points),# time_for_downsample),
                                                   axis=1)

                    except np.AxisError:
                        episode_states = downsample_regular(trimmed_state, num_points)# time_for_downsample)
                else:
                    try:
                        episode_states = np.append(episode_states, trimmed_state.drop('Time', axis=1), axis=1)
                    except np.AxisError:
                        episode_states = trimmed_state.drop('Time', axis=1)
            # This trims the data in the labels to just the parts we want to predict with the LSTM
            labels = [label_data[1], label_data[2], label_data[3], label_data[4],
                      label_data[5], label_data[6], label_data[7], label_data[10], label_data[8]]
            # labels = ['f0_proximal', 'f0_distal', 'f1_proximal', 'f1_distal',
            #           'f2_proximal', 'f2_distal', 'slip', 'success_or_failure', 'drop']

            # This looks through the data in the labels to turn strings into 1 or 0 for true or false and make sure we
            # aren't missing any data or contain any improperly labeled data
            try:
                print(labels)
                if labels[6].lower() in success_keys:
                    labels[6] = 1
                elif labels[6].lower() in fail_keys:
                    labels[6] = 0
                else:
                    raise ValueError
                if labels[7].lower() in success_keys:
                    labels[7] = 1
                elif labels[7].lower() in fail_keys:
                    labels[7] = 0
                else:
                    raise ValueError
                if labels[8].lower() in success_keys:
                    labels[8] = 1
                elif labels[8].lower() in fail_keys:
                    labels[8] = 0
                else:
                    raise ValueError
                count += 1
                episode_labels = np.zeros([len(labels), episode_states.shape[0]])
                episode_labels = episode_labels.transpose()
                episode_labels[:] = labels
            except ValueError:
                print(f'missing label or incorrect label in {big_name}{label_file}, skipping and going on to next one')
                count += 1
                continue

            # finally, we append our processed data to the final data, label and use_or_not to be used later
            if np.isnan(episode_states).any():
                print('there was a nan in the states, discarding data')
                raise TypeError
            else:
                try:
                    final_state = np.append(final_state, episode_states, axis=0)
                except NameError:
                    print('established final_state')
                    final_state = episode_states
                print('built up final state', np.shape(final_state))
                episode_times.append([episode_times[-1][1] + 1, len(final_state) - 1, folder_num])
                try:
                    final_label = np.append(final_label, episode_labels, axis=0)
                except NameError:
                    final_label = episode_labels

                try:
                    use_or_not = np.append(use_or_not, use_part, axis=0)
                except NameError:
                    use_or_not = use_part
        folder_num += 1

    # this removes the first instance of episode times which is just a placeholder
    episode_times.pop(0)
    episode_times = np.array(episode_times)
    test_inds = []
    train_inds = []
    print(episode_times[:, -1])
    # for part in range(np.max(episode_times[:, -1])):
    #     limited_times = episode_times[episode_times[:, -1] == part]
    # np.random.shuffle(limited_times)  # episode_times)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.2 * len(episode_times))
    train_inds.extend([episode_times[i][0], episode_times[i][1]] for i in range(len(episode_times) - test_portion))
    test_inds.extend([episode_times[-i - 1][0], episode_times[-i - 1][1]] for i in range(test_portion))

    # this separates the episode into train and test episodes
    train_state = [final_state[train_inds[i][0]:train_inds[i][1]] for i in range(len(train_inds))]
    train_trim_inds = [use_or_not[train_inds[i][0]:train_inds[i][1]] for i in range(len(train_inds))]
    train_label = [final_label[train_inds[i][0]:train_inds[i][1]] for i in range(len(train_inds))]
    test_state = [final_state[test_inds[i][0]:test_inds[i][1]] for i in range(len(test_inds))]
    test_trim_inds = [use_or_not[test_inds[i][0]:test_inds[i][1]] for i in range(len(test_inds))]
    test_label = [final_label[test_inds[i][0]:test_inds[i][1]] for i in range(len(test_inds))]

    train_label = np.array(unpack_arr(train_label))
    test_label = np.array(unpack_arr(test_label))
    test_state = np.array(unpack_arr(test_state))
    train_state = np.array(unpack_arr(train_state))
    train_trim_inds = np.array(unpack_arr(train_trim_inds))
    test_trim_inds = np.array(unpack_arr(test_trim_inds))

    # this picks an episode from the test state to plot later
    plot_state = final_state[train_inds[0][0]:train_inds[0][1]]
    plot_label = final_label[train_inds[0][0]:train_inds[0][1]]
    plot_inds = use_or_not[train_inds[0][0]:train_inds[0][1]]
    reduced_plot_state = plot_state[plot_inds == 1, :]
    reduced_plot_label = plot_label[plot_inds == 1, :]

    # this trims out the data before the grab/pick that we don't want to consider
    reduced_train_state = train_state[train_trim_inds == 1, :]
    reduced_test_state = test_state[test_trim_inds == 1, :]
    reduced_train_label = train_label[train_trim_inds == 1, :]
    reduced_test_label = test_label[test_trim_inds == 1, :]

    # this trims out the arm force and arm pose from the state
    # mask = list(range(27))
    # mask.extend(list(range(45,51)))
    # print(mask)
    # reduced_train_state = reduced_train_state[:,mask]
    # reduced_test_state = reduced_test_state[:,mask]

    # this saves the test state so that if we want to do additional evaluation of this trained model we can load the test
    # state and ensure that it doesn't evaluate on a point it has seen before
    np.save('test_state.npy', reduced_test_state)
    np.save('test_label.npy', reduced_test_label)
    np.save('all_state_info.npy', final_state)
    np.save('all_label_info.npy', final_label)
    np.save('all_use_info.npy', use_or_not)
    np.save('all_episode_info.npy', np.array(episode_times))
    data_file = {'train_state': train_state, 'train_label': train_label,
                 'test_state': test_state, 'test_label': test_label,
                 'train_reduce_inds': train_trim_inds, 'test_reduce_inds': test_trim_inds,
                 'train_indexes': train_inds, 'test_indexes': test_inds}
    file = open('apple_dataset.pkl', 'wb')
    pkl.dump(data_file, file)
    file.close()
    print('all files saved')
