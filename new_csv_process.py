#!/usr/bin/env python3
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
from copy import deepcopy

class GraspProcessor():
    def __init__(self):
        self.test_grasp_data = {}
        self.test_pick_data = {}
        self.validation_grasp_data = {}
        self.validation_pick_data = {}
        self.test_combined_data = {}
        self.validation_combined_data = {}
        self.csv_order = {'training_set':{'successful':[],'failed':[], 'could_be_success':[]},'test_set':{'successful':[],'failed':[], 'could_be_success':[]}}

        self.top_level = ['training_set', 'test_set']
        self.mid_level = 'pp4_folders_labeled'
        self.bot_level = ['successful', 'failed', 'could_be_success']
        self.data_labels = {'Arm Force': [0, 1, 2],
                            'Arm Torque': [3, 4, 5],
                            'IMU Acceleration': [6, 7, 8, 15, 16, 17, 24, 25, 26],
                            'IMU Gyro': [9, 10, 11, 18, 19, 20, 27, 28, 29],
                            'Finger Position': [12, 21, 30],
                            'Finger Speed': [13, 22, 31],
                            'Finger Effort': [14, 23, 32]}

    def process_full_test(self, path, validate=False):
        self.csv_order = {'training_set':{'successful':[],'failed':[], 'could_be_success':[]},'test_set':{'successful':[],'failed':[], 'could_be_success':[]}}
        self.process_data_iterable(path + '/GRASP', validate)
        self.process_data_pick(path + '/PICK', validate)
        if validate:
            combined_data = self.validation_grasp_data.copy()
            combined_data['test_state'] = np.append(self.validation_grasp_data['test_state'],self.validation_pick_data['test_state'], axis=1)
            combined_data['test_label'] = np.append(self.validation_grasp_data['test_label'],self.validation_pick_data['test_label'], axis=1)
            with open('combined_validation_dataset.pkl', 'wb') as file:
                pkl.dump(combined_data, file)
                file.close()
                print()
                print('all files saved')
                self.validation_combined_data = combined_data
        else:
            combined_data = self.test_grasp_data.copy()

            for i in range(len(combined_data['test_state'])):
                combined_data['test_state'][i].extend(self.test_pick_data['test_state'][i])
                combined_data['test_label'][i].extend(self.test_pick_data['test_label'][i])
                if combined_data['test_pick_title'][i] != self.test_pick_data['test_pick_title'][i]:
                    print('WE ARE FUCKED')
                    print(combined_data['test_pick_title'][i],self.test_pick_data['test_pick_title'][i])
            for i in range(len(combined_data['train_state'])):
                combined_data['train_state'][i].extend(self.test_pick_data['train_state'][i])
                combined_data['train_label'][i].extend(self.test_pick_data['train_label'][i]) 
                if combined_data['train_pick_title'][i] != self.test_pick_data['train_pick_title'][i]:
                    print('WE ARE FUCKED')
                    print(combined_data['train_pick_title'][i],self.test_pick_data['train_pick_title'][i])
            with open('combined_train_test_dataset.pkl', 'wb') as file:
                pkl.dump(combined_data, file)
                file.close()
                print()
                print('all files saved')
                self.test_combined_data = combined_data
        self.save_csv('gim')

    def process_data_pick(self, path, validate=False):
        """
        Reads apple picking data saved in the given path, generates labels for all picks and saves
        processed data as npy files for later use.
        @param path - Filepath containing rosbags and csvs from apple picking"""
        pcount = 0
        ncount = 0
        states = {'training_set': [], 'test_set': []}
        labels = {'training_set': [], 'test_set': []}
        final_csv_order = {'training_set':{'successful':[],'failed':[], 'could_be_success':[]},'test_set':{'successful':[],'failed':[], 'could_be_success':[]}}
        pick_names = {'training_set': [], 'test_set': []}
        # This loop iterates through all the folders in the path folder, which contains different folders for each set of starting noises
        for top_folder in self.top_level:
            for folder in self.bot_level:
                if len(self.csv_order[top_folder][folder]) == 0:
                    self.csv_order[top_folder][folder] = os.listdir(path + '/' + self.mid_level + '/' + top_folder + '/' + folder)
                    for i in range(len(self.csv_order[top_folder][folder])):
                        self.csv_order[top_folder][folder][i] = self.csv_order[top_folder][folder][i]
                # This loop iterates through the folders in the selected folder in path. These folders contain csvs with all the
                # episode data and have the same pick number as the csv label file in the folder above
                for data_file_name in self.csv_order[top_folder][folder]:
                    real_file_name = data_file_name.replace('grasp','pick')
                    episode_state = []
                    print('Opening up name', real_file_name)
                    temp = data_file_name.split('_')
                    if temp[2] == 'pick':
                        key_thing = 'pick' + temp[3]
                    else:
                        key_thing = temp[2]
                    with open(path + '/'  + self.mid_level + '/' + top_folder + '/' + folder + '/' + real_file_name, 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        temp = False
                        for row in reader:
                            if temp:
                                episode_state.append(list(row[1:]))
                            else:
                                temp = True
                    if len(episode_state) > 0:
                        e_len =len(episode_state)
                        episode_state = np.array(episode_state)
                        episode_state = episode_state.astype(float)
                        episode_state = episode_state.tolist()
                        if folder == 'successful':
                            episode_label = [1] * int(e_len)
                            labels[top_folder].append(episode_label)
                            pcount += 1
                        elif folder == 'failed':
                            episode_label = [0] * int(e_len)
                            labels[top_folder].append(episode_label)
                            ncount += 1
                        else:
                            episode_label = [1] * int(e_len)
                            labels[top_folder].append(episode_label)
                            pcount += 1#                            print(top_folder)
#                        input(episode_state.copy())
                        states[top_folder].append(episode_state.copy())
                        final_csv_order[top_folder][folder].append(data_file_name)
                        pick_names[top_folder].append([[key_thing]])

        if self.csv_order != final_csv_order:
            print('pick csv order issue, original csv order: ', self.csv_order, 'new order: ', final_csv_order)
#        print(type(states['training_set']))
#        input(states['training_set'][0])
        data_file = {'train_state': states['training_set'], 'train_label': labels['training_set'],
                     'test_state': states['test_set'], 'test_label': labels['test_set'],
                     'train_pick_title': pick_names['training_set'], 'test_pick_title':pick_names['test_set']}
#        input(states['training_set'])
        if validate:
            file=open('validation_pick_dataset.pkl','wb')
            self.validation_pick_data = data_file.copy()
        else:
            file = open('train_test_pick_dataset.pkl', 'wb')
            self.test_pick_data = data_file.copy()
        pkl.dump(data_file, file)
        file.close()
        print()
        print('all files saved')

    def process_data_iterable(self, path, validate=False):
        """
        Reads apple picking data saved in the given path, generates labels for all picks and saves
        processed data as npy files for later use.
        @param path - Filepath containing rosbags and csvs from apple picking"""
        pcount = 0
        ncount = 0
        states = {'training_set': [], 'test_set': []}
        labels = {'training_set': [], 'test_set': []}
        pick_names = {'training_set': [], 'test_set': []}
        # This loop iterates through all the folders in the path folder,
        # which contains different folders for each set of starting noises
        final_csv_order = {'training_set':{'successful':[],'failed':[], 'could_be_success':[]},'test_set':{'successful':[],'failed':[], 'could_be_success':[]}}
        for top_folder in self.top_level:
            for folder in self.bot_level:
                if len(self.csv_order[top_folder][folder]) == 0:
                    self.csv_order[top_folder][folder] = os.listdir(path + '/'  + self.mid_level + '/' + top_folder + '/' + folder)
                    for i in range(len(self.csv_order[top_folder][folder])):
                        self.csv_order[top_folder][folder][i] = self.csv_order[top_folder][folder][i]
                # This loop iterates through the folders in the selected folder in path. These folders contain csvs with all the
                # episode data and have the same pick number as the csv label file in the folder above
                
                for data_file_name in self.csv_order[top_folder][folder]:
                    episode_state = []
                    print('Opening up name', data_file_name)
                    temp = data_file_name.split('_')
                    if temp[2] == 'pick':
                        key_thing = 'pick' + temp[3]
                    else:
                        key_thing = temp[2]
                    with open(path + '/'  + self.mid_level + '/' + top_folder + '/' + folder + '/' + data_file_name, 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        temp = False
                        for row in reader:
                            if temp:
                                episode_state.append(list(row[1:]))
                            else:
                                temp = True
                    if len(episode_state) > 0:
                        e_len =len(episode_state)
                        episode_state = np.array(episode_state)
                        episode_state = episode_state.astype(float)
                        episode_state = episode_state.tolist()
                        if folder == 'successful':
                            episode_label = [1] * int(e_len)
                            labels[top_folder].append(episode_label)
                            pcount += 1
                        elif folder == 'failed':
                            episode_label = [0] * int(e_len)
                            labels[top_folder].append(episode_label)
                            ncount += 1
                        else:
                            episode_label = [1] * int(e_len)
                            labels[top_folder].append(episode_label)
                            pcount += 1#                            print(top_folder)
#                        input(episode_state.copy())
#                    if len(episode_state) > 0:
#                        e_len = len(episode_state)
#                        episode_state = np.array(episode_state)
#                        episode_state = episode_state.astype(float)
#                        episode_state = episode_state.tolist()
#                        episode_label = [0.5] * int(e_len/4)
#                        if folder == 'successful':
#                            middle_label = np.array(range(int(e_len/2))) / int(e_len/2) * 0.5 + 0.5
#                            episode_label.extend(middle_label)
#                            episode_label.extend([1] * int(e_len-int(e_len/4)-int(e_len/2)))
#                            labels[top_folder].append(episode_label)
#                            pcount += 1
#                        elif folder == 'failed':
#                            middle_label = np.array(range(int(e_len/2))) / int(e_len/2) * -0.5 + 0.5
#                            episode_label.extend(middle_label)
#                            episode_label.extend([0] * int(e_len-int(e_len/4)-int(e_len/2)))
#                            labels[top_folder].append(episode_label)
#                            ncount += 1
#                        else:
#                            middle_label = np.array(range(int(e_len/2))) / int(e_len/2) * 0.5 + 0.5
#                            episode_label.extend(middle_label)
#                            episode_label.extend([1] * int(e_len-int(e_len/4)-int(e_len/2)))
#                            labels[top_folder].append(episode_label)
#                            print(top_folder)
                            
                            
                        states[top_folder].append(episode_state.copy())
                        pick_names[top_folder].append([[key_thing]])
                        final_csv_order[top_folder][folder].append(data_file_name)
        if self.csv_order != final_csv_order:
            print('grasp csv order issue, original csv order: ',self.csv_order, 'new order: ', final_csv_order)
        data_file = {'train_state': states['training_set'], 'train_label': labels['training_set'],
                     'test_state': states['test_set'], 'test_label': labels['test_set'],
                     'train_pick_title': pick_names['training_set'], 'test_pick_title':pick_names['test_set']}
#        input(states['training_set'])
        if validate:
            file=open('validation_grasp_dataset.pkl','wb')
            self.validation_grasp_data = data_file.copy()
        else:
            file = open('train_test_grasp_dataset.pkl', 'wb')
            self.test_grasp_data = data_file.copy()
        pkl.dump(data_file, file)
        file.close()
        print()
        print('all files saved')
        self.save_csv('gim')

    def save_csv(self, filename, dataset_name=None):
        
        #TODO: Make this a way to actually grab the right set of data instead of a cobbled together mess
        if dataset_name is None:
            dataset_name = 'combined_bill'
        if dataset_name == 'combined_test':
            dataset = self.test_grasp_data
            print([key for key in dataset.keys()])
            with open('train_test_grasp.csv','w+') as file:
                z = csv.writer(file)
                for episode_state, episode_label in zip(dataset['train_state'], dataset['train_label']):
                    for timestep_state, timestep_label in zip(episode_state,episode_label):
                        temp = timestep_state.copy()
                        temp.extend([timestep_label])
                        z.writerow(temp)
        elif dataset_name == 'combined_bill':
#            dataset = self.test_grasp_data
#            with open('proxy_train_grasp_bill.csv','w+') as file:
#                z = csv.writer(file)
#                for episode_state, episode_label in zip(dataset['train_state'], dataset['train_label']):
#                    temp = []
#                    for timestep_state, timestep_label in zip(episode_state,episode_label):
#                        temp.extend(timestep_state)
#                    temp.extend([timestep_label])
#                    z.writerow(temp)
#            with open('proxy_test_grasp_bill.csv','w+') as file:
#                z = csv.writer(file)
#                for episode_state, episode_label in zip(dataset['test_state'], dataset['test_label']):
#                    temp = []
#                    for timestep_state, timestep_label in zip(episode_state,episode_label):
#                        temp.extend(timestep_state)
#                    temp.extend([timestep_label])
#                    z.writerow(temp)            
            dataset = self.validation_grasp_data
            print([key for key in dataset.keys()])
            with open('proxy_triple_check.csv','w+') as file:
                z = csv.writer(file)
                for episode_state, episode_label in zip(dataset['test_state'], dataset['test_label']):
                    temp = []
                    for timestep_state, timestep_label in zip(episode_state,episode_label):
                        temp.extend(timestep_state[0:3])
                    temp.extend([timestep_label])
                    z.writerow(temp)
        input('did i do well?')