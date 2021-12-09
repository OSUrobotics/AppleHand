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


class RNNDataset(IterableDataset):
    def __init__(self, state_list, label_list, batch_size):
        self.state_list = state_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.shape = (len(self.state_list), len(self.state_list[0]), len(self.state_list[0][0]))

    def divide_into_batches(self, state, label):
        for i in range(0, len(state), self.batch_size):
            yield state[i:i + self.batch_size], label[i:i + self.batch_size]

    @property
    def shuffled_episodes(self):
        shuffled_state, shuffled_label = zip(*random.sample(list(zip(self.state_list, self.label_list)), len(self.state_list)))
        batched_data = list(self.divide_into_batches(shuffled_state, shuffled_label))
        shuffled_state = [np.concatenate(arr_list[0]) for arr_list in batched_data]
        shuffled_label = [np.concatenate(arr_list[1]) for arr_list in batched_data]
        return shuffled_state, shuffled_label

    def get_episodes(self):
        states, labels = self.shuffled_episodes
        state_iter = iter(states)
        label_iter = iter(labels)
        return zip(state_iter, label_iter)

    def __iter__(self):
        return self.get_episodes()

    def __len__(self):
        return len(self.state_list)


def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    @param long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
