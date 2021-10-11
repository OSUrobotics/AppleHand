#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:13:18 2021

@author: orochi
"""
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle as pkl
import datetime
from AppleClassifier import AppleClassifier


def perform_ablation(train_dataset, test_dataset, args):
    """
    Function to perform an ablation study to determine which feature is the
    most important for the classification
    @param train_dataset - a TensorDataset containing the training data
    @param test_dataset - a TensorDataset containing the testing data
    @param args - an argparse.Namespace containing hyperparameters for
    the classifiers
    @return - a dictionary containing the acuracy, standard deviation, number
    of inputs and removed features for every step of the ablation"""
    labels = {'IMU Accelearation': [0, 1, 2, 9, 10, 11, 18, 19, 20], 
              'IMU Velocity': [3, 4, 5, 12, 13, 14, 21, 22, 23],
              'Joint Pos': [6, 15, 24],
              'Joint Velocity': [7, 16, 25],
              'Joint Effort': [8, 17, 26],
              'Arm Joint State': [27, 28, 29, 30, 31, 32],
              'Arm Joint Velocity': [33, 34, 35, 36, 37, 38],
              'Arm Joint Effort': [39, 40, 41, 42, 43, 44],
              'Wrench Force': [45, 46, 47],
              'Wrench Torque': [48, 49, 50]}
    full_list = np.array(range(51))
    missing_labels = []
    missing_names = ''
    worst_names = []
    sizes = [51]

    full_train_loader = DataLoader(train_dataset, shuffle=False,
                                   batch_size=args.batch_size, drop_last=True)
    full_test_loader = DataLoader(test_dataset, shuffle=False,
                                  batch_size=args.batch_size, drop_last=True)
    performance = []
    for i in range(3):
        full_lstm = AppleClassifier(full_train_loader, full_test_loader,
                                    vars(args))
        full_lstm.train()
        performance = np.max(full_lstm.accuracies)
        print('model finished, saving now')
        full_lstm.save_data()
    best_accuracies = [np.average(performance)]
    best_acc_std_dev = np.std(best_accuracies)
    for phase in range(9):
        feature_combo_accuracies = []
        names = []
        indexes = []
        acc_std_dev = []
        for name, missing_label in labels.items():
            temp = np.ones(51, dtype=bool)
            temp[missing_label] = False
            try:
                temp[missing_labels] = False
            except:
                pass
            used_labels = full_list[temp]
            reduced_train_dataset = TensorDataset(train_dataset.tensors[0][:, used_labels],
                                                             train_dataset.tensors[1])
            reduced_test_dataset = TensorDataset(test_dataset.tensors[0][:, used_labels],
                                                             train_dataset.tensors[1])
            reduced_train_loader = DataLoader(reduced_train_dataset, shuffle=False,
                                              batch_size=args.batch_size, drop_last=True)
            reduced_test_loader = DataLoader(reduced_test_dataset, shuffle=False,
                                             batch_size=args.batch_size, drop_last=True)
            performance = []
            for i in range(3):
                base_lstm = AppleClassifier(reduced_train_loader,
                                            reduced_test_loader, vars(args))
                base_lstm.train()
                performance.append(np.max(base_lstm.accuracies))
                print('model finished, saving now')
                base_lstm.save_data()
            feature_combo_accuracies.append(np.average(performance))
            acc_std_dev.append(np.std(performance))
            names.append(name)
            indexes.append(missing_label)
        best_one = np.argmax(feature_combo_accuracies)
        print('best combination', best_one, np.shape(feature_combo_accuracies), np.shape(names))
        missing_names = missing_names + names[best_one]
        missing_labels.extend(indexes[best_one])
        best_accuracies.append(feature_combo_accuracies[best_one])
        best_acc_std_dev.append(acc_std_dev[best_one])
        worst_names.append(names[best_one])
        sizes.append(sizes[phase] - len(labels[names[best_one]]))
        labels.pop(names[best_one])
    print('ablation finished. best accuracies throughout were', best_accuracies)
    print('names removed in this order', worst_names)

    grasp_ablation_dict = {'num inputs': sizes, 'best accuracy': best_accuracies,
                           'std dev': best_acc_std_dev, 'names': worst_names}
    file = open('grasp_ablation_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M")
                + '.pkl', 'wb')
    pkl.dump(grasp_ablation_dict, file)
    file.close()
    return grasp_ablation_dict
