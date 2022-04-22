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
import torch



class RNNDataset(Dataset):
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
            self.episodes.append({'state': torch.tensor(state), 'label': torch.tensor(label), 'name': name})
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


def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
