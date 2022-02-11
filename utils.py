#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:54:09 2021

@author: Nigel Swenson
"""

from torch.utils.data import IterableDataset
import random
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

class RNNDataset(IterableDataset):
    def __init__(self, state_list, label_list, batch_size, range_params=None):
        
        self.label_list = label_list
        self.batch_size = batch_size
        
        try:
            if range_params == False:
                print('WE ARENT NORMALIZING YOU DUMMY')
                self.state_list = state_list.copy()
                self.range_params = None
                self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))
            else:
                self.state_list, self.range_params = self.scale(state_list.copy(),range_params)
                self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))
#            print('shape',self.shape)
        except IndexError:
            print('Database is None, building a dummy dataset')
            self.shape = (0,0,0)
            self.state_list = state_list
        except  ValueError:
            print('Database is None, not scaling it and building a dummy dataset')
            self.shape=(0,0,0)
            self.state_list = state_list
#        print(type(state_list))
    def divide_into_batches(self, state, label):
        for i in range(0, len(state), self.batch_size):
            yield state[i:i + self.batch_size], label[i:i + self.batch_size]

    @property
    def shuffled_episodes(self):
        shuffled_state, shuffled_label = zip(*random.sample(list(zip(self.state_list, self.label_list)), len(self.state_list)))
        batched_data = list(self.divide_into_batches(shuffled_state, shuffled_label))
        shuffled_state = [np.concatenate(arr_list[0]) for arr_list in batched_data]
        shuffled_label = [np.concatenate(arr_list[1]) for arr_list in batched_data]
        lens = []
        for arr_list in batched_data:
#            print(len(arr_list[0]))
            temp = [len(arr_list[0][i]) for i in range(len(arr_list[0]))]
            lens.append(temp)
#        input(lens)
        return shuffled_state, shuffled_label, lens

    def get_episodes(self):
        states, labels, lens = self.shuffled_episodes
        state_iter = iter(states)
        label_iter = iter(labels)
        return zip(state_iter, label_iter, lens)

    def get_params(self):
        return self.range_params

    def __iter__(self):
        return self.get_episodes()

    def __len__(self):
        return len(self.state_list)

    @staticmethod
    def scale(unscaled_list, range_params=None):
#        print('normalizing the parameters!')
        #warning, this will be ugly because i am making it with the idea that the sublists are of arbitrary length
        if range_params is None:
#            print('finding new range params')
            range_params={'top':[],'bot':[]}
            just_datapoints = unpack_arr(unscaled_list)
#            print(just_datapoints)
            range_params['top'] = np.max(just_datapoints, axis=0)
            range_params['bot'] = np.min(just_datapoints, axis=0)
    
        for i in range(len(unscaled_list)):
            for j in range(len(unscaled_list[i])):
                for k in range(len(unscaled_list[i][j])):
                    unscaled_list[i][j][k] = (unscaled_list[i][j][k] - range_params['bot'][k])/ (range_params['top'][k]-range_params['bot'][k]) 
        return unscaled_list, range_params

def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    @param long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
