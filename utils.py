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
    def __init__(self, data_list, label_list, batch_size):
        self.data_list = data_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.shape = (len(self.data_list), len(self.data_list[0]), len(self.data_list[0][0]))

    @property
    def shuffled_episodes(self):
        shuffled_state, shuffled_label = zip(*random.sample(list(zip(self.data_list, self.label_list)), len(self.data_list)))
        return shuffled_state, shuffled_label

    def get_episodes(self):
        states, labels = self.shuffled_episodes
        state_iter = iter(states)
        label_iter = iter(labels)
        return zip(state_iter, label_iter)

    def __iter__(self):
        return self.get_episodes()

    def __len__(self):
        return len(self.data_list)