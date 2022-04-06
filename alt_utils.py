#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:54:09 2021

@author: Nigel Swenson
"""

from torch.utils.data import IterableDataset, Dataset
import random
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import time



class RNNSamplerDataset(Dataset):
    def __init__(self, state_list, label_list, episode_names, batch_size, range_params=None):
        """
        Class to contain states and labels for recurrent neural networks made of collection of episodes
        :param state_list: list of episodes containing state data
        :param label_list: list of episodes containing labels for each timestep
        :param episode_names: name of every episode
        :param batch_size: number of episodes returned in a single batch
        :param range_params: min and max for each parameter to normalize data
        """
        self.batch_size = batch_size
        self.change_success_rate = False
        self.episodes = []
        for state, label, name in zip(state_list, label_list, episode_names):
            self.episodes.append({'state': state, 'label': label, 'name': name})
        self.time_getting_eps = 0
        self.eps = 0
        try:
            if not range_params:
                print('WE ARENT NORMALIZING YOU DUMMY')
                self.state_list = state_list.copy()
                self.range_params = None
                self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))
            else:
                self.state_list, self.range_params = self.scale(state_list.copy(), range_params)
                self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))
        except IndexError:
            print('Database is None, building a dummy dataset')
            self.shape = (0, 0, 0)
            self.state_list = state_list
        except ValueError:
            print('Database is None, not scaling it and building a dummy dataset')
            self.shape = (0, 0, 0)
            self.state_list = state_list

    def get_params(self):
        return self.range_params

    def __getitem__(self, index):
        return self.episodes[index]['state'], self.episodes[index]['label'], len(self.episodes[index]['state']), self.episodes[index]['name']

    def __len__(self):
        return len(self.state_list)

    @staticmethod
    def scale(unscaled_list, range_params=None):
        """
        scales state data using range params if given, if not finds range params and scales state data
        """
        # warning, this will be ugly because i am making it with the idea that the sublists are of arbitrary length
        if range_params is None:
            range_params = {'top': [], 'bot': []}
            just_datapoints = unpack_arr(unscaled_list)
            range_params['top'] = np.max(just_datapoints, axis=0)
            range_params['bot'] = np.min(just_datapoints, axis=0)

        for i in range(len(unscaled_list)):
            for j in range(len(unscaled_list[i])):
                for k in range(len(unscaled_list[i][j])):
                    unscaled_list[i][j][k] = (unscaled_list[i][j][k] - range_params['bot'][k]) / (
                            range_params['top'][k] - range_params['bot'][k])
        return unscaled_list, range_params

class RNNDataset(IterableDataset):
    def __init__(self, state_list, label_list, episode_names, batch_size, range_params=None):
        """
        Class to contain states and labels for recurrent neural networks made of collection of episodes
        :param state_list: list of episodes containing state data
        :param label_list: list of episodes containing labels for each timestep
        :param episode_names: name of every episode
        :param batch_size: number of episodes returned in a single batch
        :param range_params: min and max for each parameter to normalize data
        """
        self.batch_size = batch_size
        self.change_success_rate = False
        self.episodes = []
        for state, label, name in zip(state_list, label_list, episode_names):
            self.episodes.append({'state': state, 'label': label, 'name': name})
        self.time_getting_eps = 0
        self.eps = 0
        try:
            if not range_params:
                print('WE ARENT NORMALIZING YOU DUMMY')
                self.state_list = state_list.copy()
                self.range_params = None
                self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))
            else:
                self.state_list, self.range_params = self.scale(state_list.copy(), range_params)
                self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))
        except IndexError:
            print('Database is None, building a dummy dataset')
            self.shape = (0, 0, 0)
            self.state_list = state_list
        except ValueError:
            print('Database is None, not scaling it and building a dummy dataset')
            self.shape = (0, 0, 0)
            self.state_list = state_list

    def divide_into_batches(self, state, label, names):
        """
        splits state, label and names into batches based on batch size
        """
        for i in range(0, len(state), self.batch_size):
            yield state[i:i + self.batch_size], label[i:i + self.batch_size], names[i:i + self.batch_size]

    @property
    def shuffled_episodes(self):
        """
        function to shuffle all episodes in dataset
        :return: shuffled states, labels, lengths of episodes, and names of episodes
        """
        if self.change_success_rate:
            
            shuffled_state = [temp['state'] for temp in self.episodes]
            shuffled_label = [temp['label'] for temp in self.episodes]
            shuffled_name = [temp['name'] for temp in self.episodes]
        else:
            np.random.shuffle(self.episodes)
            shuffled_state = [temp['state'] for temp in self.episodes]
            shuffled_label = [temp['label'] for temp in self.episodes]
            shuffled_name = [temp['name'] for temp in self.episodes]
        batched_data = list(self.divide_into_batches(shuffled_state, shuffled_label, shuffled_name))
        shuffled_state = [np.concatenate(arr_list[0]) for arr_list in batched_data]
        shuffled_label = [np.concatenate(arr_list[1]) for arr_list in batched_data]
        names = [np.concatenate(arr_list[2]) for arr_list in batched_data]
        lens = []
        for arr_list in batched_data:
            temp = [len(arr_list[0][i]) for i in range(len(arr_list[0]))]
            lens.append(temp)
        return shuffled_state, shuffled_label, lens, names

    def get_episodes(self):
        """
        current time is 0.0639 (seems new method slightly better)
        produces zip for iter function of shuffled episode data
        """
        t0 = time.time()
        states, labels, lens, names = self.shuffled_episodes
        state_iter = iter(states)
        label_iter = iter(labels)
        t1 = time.time()
        self.time_getting_eps += t1 - t0
        self.eps += 1
        if self.eps % 10 == 1:
            self.print_times()
            print(names[0])
        return zip(state_iter, label_iter, lens, names)

    def get_params(self):
        return self.range_params

    def __iter__(self):
        return self.get_episodes()

    def __len__(self):
        return len(self.state_list)

    def print_times(self):
        print(self.time_getting_eps / self.eps)

    @staticmethod
    def scale(unscaled_list, range_params=None):
        """
        scales state data using range params if given, if not finds range params and scales state data
        """
        # warning, this will be ugly because i am making it with the idea that the sublists are of arbitrary length
        if range_params is None:
            range_params = {'top': [], 'bot': []}
            just_datapoints = unpack_arr(unscaled_list)
            range_params['top'] = np.max(just_datapoints, axis=0)
            range_params['bot'] = np.min(just_datapoints, axis=0)

        for i in range(len(unscaled_list)):
            for j in range(len(unscaled_list[i])):
                for k in range(len(unscaled_list[i][j])):
                    unscaled_list[i][j][k] = (unscaled_list[i][j][k] - range_params['bot'][k]) / (
                            range_params['top'][k] - range_params['bot'][k])
        return unscaled_list, range_params



def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
