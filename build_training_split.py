#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:06:23 2022

@author: orochi
"""

import os
import numpy as np
import re
import shutil


def generate_training_split(num_batches, path=None):
    if path is None:
        path = './raw_data/RAL22_Paper/6_real_fall21_x5/'
    if type(num_batches) is not int:
        raise Exception('what the actual fuck dude, give me an integer')

    # load in the names of all the files we are working with
    grasp_successes = os.listdir(path + 'GRASP/new_pp5_labeled/test_set/successful')
    grasp_fails = os.listdir(path + 'GRASP/new_pp5_labeled/test_set/failed')
    grasp_successes.extend(os.listdir(path + 'GRASP/new_pp5_labeled/test_set/could_be_success'))
    pick_successes = os.listdir(path + 'PICK/new_pp5_labeled/test_set/successful')
    pick_fails = os.listdir(path + 'PICK/new_pp5_labeled/test_set/failed')
    pick_successes.extend(os.listdir(path + 'PICK/new_pp5_labeled/test_set/could_be_success'))

    # shuffle the successes and failes and split them into batches
    np.random.shuffle(grasp_successes)
    np.random.shuffle(grasp_fails)
    grasp_success_batches = np.array_split(grasp_successes, num_batches)
    grasp_fail_batches = np.array_split(grasp_fails, num_batches)

    # find which numbers we are using in the shuffled batches so that we get the same pick batches
    pick_success_temp = []
    for thing in grasp_successes:
        temp = re.search('_\d+_', thing)
        for name in pick_successes:
            if temp.group(0) in name:
                pick_success_temp.append(name)
                break
    pick_successes = pick_success_temp
    pick_success_batches = np.array_split(pick_successes, num_batches)

    pick_fail_temp = []
    for thing in grasp_fails:
        temp = re.search('_\d+_', thing)
        for name in pick_fails:
            if temp.group(0) in name:
                pick_fail_temp.append(name)
                break
    pick_fails = pick_fail_temp
    pick_fail_batches = np.array_split(pick_fails, num_batches)

    # Set up the file tree in the way we expect it in csv_process
    os.makedirs(path + 'training_testing_split/set1/GRASP/new_pp5_labeled/test_set/failed/')
    os.makedirs(path + 'training_testing_split/set1/GRASP/new_pp5_labeled/test_set/successful/')
    os.makedirs(path + 'training_testing_split/set1/GRASP/new_pp5_labeled/test_set/could_be_success/')
    shutil.copytree(path + 'training_testing_split/set1/GRASP/new_pp5_labeled/test_set/',
                    path + 'training_testing_split/set1/GRASP/new_pp5_labeled/training_set/')
    shutil.copytree(path + 'training_testing_split/set1/GRASP/', path + 'training_testing_split/set1/PICK/')
    for i in range(num_batches - 1):
        shutil.copytree(path + 'training_testing_split/set1', path + 'training_testing_split/set' + str(i + 2) + '/')

    # copy the files listed in the batches to their apropriate file path
    batch_nums = list(range(1, num_batches + 1))
    for set_num in batch_nums:
        other_sets = [x for x in batch_nums if x != set_num]
        for file in pick_fail_batches[set_num - 1]:
            shutil.copyfile(path + 'PICK/new_pp5_labeled/test_set/failed/' + file,
                            path + 'training_testing_split/set' + str(
                                set_num) + '/PICK/new_pp5_labeled/test_set/failed/' + file)
            for o_set in other_sets:
                shutil.copyfile(path + 'PICK/new_pp5_labeled/test_set/failed/' + file,
                                path + 'training_testing_split/set' + str(
                                    o_set) + '/PICK/new_pp5_labeled/training_set/failed/' + file)
        for file in grasp_fail_batches[set_num - 1]:
            shutil.copyfile(path + 'GRASP/new_pp5_labeled/test_set/failed/' + file,
                            path + 'training_testing_split/set' + str(
                                set_num) + '/GRASP/new_pp5_labeled/test_set/failed/' + file)
            for o_set in other_sets:
                shutil.copyfile(path + 'GRASP/new_pp5_labeled/test_set/failed/' + file,
                                path + 'training_testing_split/set' + str(
                                    o_set) + '/GRASP/new_pp5_labeled/training_set/failed/' + file)
        for file in pick_success_batches[set_num - 1]:
            try:
                shutil.copyfile(path + 'PICK/new_pp5_labeled/test_set/successful/' + file,
                                path + 'training_testing_split/set' + str(
                                    set_num) + '/PICK/new_pp5_labeled/test_set/successful/' + file)
                for o_set in other_sets:
                    shutil.copyfile(path + 'PICK/new_pp5_labeled/test_set/successful/' + file,
                                    path + 'training_testing_split/set' + str(
                                        o_set) + '/PICK/new_pp5_labeled/training_set/successful/' + file)
            except:
                shutil.copyfile(path + 'PICK/new_pp5_labeled/test_set/could_be_success/' + file,
                                path + 'training_testing_split/set' + str(
                                    set_num) + '/PICK/new_pp5_labeled/test_set/successful/' + file)
                for o_set in other_sets:
                    shutil.copyfile(path + 'PICK/new_pp5_labeled/test_set/could_be_success/' + file,
                                    path + 'training_testing_split/set' + str(
                                        o_set) + '/PICK/new_pp5_labeled/training_set/successful/' + file)

        for file in grasp_success_batches[set_num - 1]:
            try:
                shutil.copyfile(path + 'GRASP/new_pp5_labeled/test_set/successful/' + file,
                                path + 'training_testing_split/set' + str(
                                    set_num) + '/GRASP/new_pp5_labeled/test_set/successful/' + file)
                for o_set in other_sets:
                    shutil.copyfile(path + 'GRASP/new_pp5_labeled/test_set/successful/' + file,
                                    path + 'training_testing_split/set' + str(
                                        o_set) + '/GRASP/new_pp5_labeled/training_set/successful/' + file)
            except:
                shutil.copyfile(path + 'GRASP/new_pp5_labeled/test_set/could_be_success/' + file,
                                path + 'training_testing_split/set' + str(
                                    set_num) + '/GRASP/new_pp5_labeled/test_set/successful/' + file)
                for o_set in other_sets:
                    shutil.copyfile(path + 'GRASP/new_pp5_labeled/test_set/could_be_success/' + file,
                                    path + 'training_testing_split/set' + str(
                                        o_set) + '/GRASP/new_pp5_labeled/training_set/successful/' + file)

    print('done, go check the folders')
    print(grasp_success_batches)


generate_training_split(2)
