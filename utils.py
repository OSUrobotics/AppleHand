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
#import matplotlib.pyplot as plt


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
        self.state_lens = [len(state) for state in state_list]
        
        try:
            if range_params == False:
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

        padded_state_list = np.zeros((len(self.state_list),max(self.state_lens),len(self.state_list[0][0])))
        padded_labels = np.ones((len(self.state_list),max(self.state_lens))) * 2

        for i, x_len in enumerate(self.state_lens):
            sequence = self.state_list[i]
            label_sequence = label_list[i]
            padded_state_list[i, 0:x_len] = sequence[:x_len]
            padded_labels[i, 0:x_len] = label_sequence[:x_len]
        for state, label, name, unpacked_len in zip(padded_state_list, padded_labels, episode_names,self.state_lens):
            self.episodes.append({'state': torch.tensor(state), 'label': torch.tensor(label), 'name': name, 'length': unpacked_len})
        self.time_getting_eps = 0
        self.eps = 0

    def get_params(self):
        return self.range_params

    def __getitem__(self, index):
        return self.episodes[index]['state'], self.episodes[index]['label'], self.episodes[index]['length'], self.episodes[index]['name']

    def __len__(self):
        return len(self.state_list)

    @staticmethod
    def scale(unscaled_list, range_params=None):
        """
        scales state data using range params if given, if not finds range params and scales state data
        """
        # warning, this will be ugly because i am making it with the idea that the sublists are of arbitrary length
#        print('not right now')
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

    @staticmethod
    def normalize(unscaled_list):
        return torch.nn.functional.normalize(unscaled_list,dim=2)

    def to(self,device):
        for ep in self.episodes:
            ep['state'] = ep['state'].to(device)
            ep['label'] = ep['label'].to(device)
        print('double checking. device should be cuda')
        print(self.episodes[0]['state'].device)
def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
