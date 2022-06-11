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
from utils import RNNDataset
import matplotlib.pyplot as plt
from copy import deepcopy


#TODO: make this work with the new sampler datasets
def perform_ablation(database, args, test=None):
    """
    Function to perform an ablation study to determine which feature is the
    most important for the classification
    @param train_dataset - a TensorDataset containing the training data
    @param validation_dataset - a TensorDataset containing the validationing data
    @param args - an argparse.Namespace containing hyperparameters for
    the classifiers
    @return - a dictionary containing the acuracy, standard deviation, number
    of inputs and removed features for every step of the ablation"""
#    labels = {'IMU Accelearation': [0, 1, 2, 9, 10, 11, 18, 19, 20], 
#              'IMU Velocity': [3, 4, 5, 12, 13, 14, 21, 22, 23],
#              'Joint Pos': [6, 15, 24],
#              'Joint Velocity': [7, 16, 25],
#              'Joint Effort': [8, 17, 26],
#              'Arm Joint State': [27, 28, 29, 30, 31, 32],
#              'Arm Joint Velocity': [33, 34, 35, 36, 37, 38],
#              'Arm Joint Effort': [39, 40, 41, 42, 43, 44],
#              'Wrench Force': [45, 46, 47],
#              'Wrench Torque': [48, 49, 50]}
#    labels = {'Arm Force': [0, 1, 2, 3], 
#              'Arm Torque': [4, 5, 6, 7],
#              'IMU Time': [8, 20, 32],
#              'IMU Acceleration': [9, 10, 11, 12, 21, 22, 23, 24, 33, 34, 35, 36],
#              'IMU Gyro': [13, 14, 15, 25, 26, 27, 37, 38, 39],
#              'Finger State Time': [16, 28, 40],
#              'Finger Position': [17, 29, 41],
#              'Finger Speed': [18, 30, 42],
#              'Finger Effort': [19, 31, 43]}
#    labels = {'Arm Force': [0, 1, 2], 
#              'Arm Torque': [3, 4, 5],
#              'IMU Acceleration': [6, 7, 8, 15, 16, 17, 24, 25, 26],
#              'IMU Gyro': [9, 10, 11, 18, 19, 20, 27, 28, 29],
#              'Finger Position': [12, 21, 30],
#              'Finger Speed': [13, 22, 31],
#              'Finger Effort': [14, 23, 32]}
    labels = {'Arm Force X': [0],
              'Arm Force Y': [1],
              'Arm Force Z': [2],
              'Arm Torque Roll': [3],
              'Arm Torque Pitch': [4],
              'Arm Torque Yaw': [5],
              'IMU Acceleration X':[6, 15, 24],
              'IMU Acceleration Y':[7, 16, 25],
              'IMU Acceleration Z':[8, 17, 26],
              'IMU Gyro X': [9, 18, 27],
              'IMU Gyro Y': [10, 19, 28],
              'IMU Gyro Z': [11, 20, 29],
              'Finger Position': [12, 21, 30],
              'Finger Speed': [13, 22, 31],
              'Finger Effort': [14, 23, 32]}
    full_list = np.array(range(33))
    missing_labels = []
    missing_names = ''
    worst_names = []
    sizes = [33]
    print([key for key in database.keys()])
    full_train_loader = RNNDataset(database['train_state'], database['train_label'], database['train_pick_title'], args.batch_size, range_params=False)
    full_validation_loader = RNNDataset(database['validation_state'], database['validation_label'], database['validation_pick_title'], args.batch_size, range_params=False)
    if test is not None:
        test_loader = RNNDataset(test['validation_state'], test['validation_label'], test['test_pick_title'], args.batch_size, range_params=False)
    else:
        test_loader = None
    performance = {'auc': [], 'acc': []}
    for i in range(3):
        full_lstm = AppleClassifier(full_train_loader, full_validation_loader, vars(args), test_dataset=test_loader)
        full_lstm.train()
        roc_max_acc = []
        max_acc = full_lstm.get_best_performance(None)
        max_auc = np.max(full_lstm.AUC)
        performance['auc'].append(max_auc)
        performance['acc'].append(np.max(max_acc))
        print('model finished, saving now.')
        full_lstm.save_data()
        print(performance)
    best_accuracies = [np.average(performance['acc'])]
    best_acc_std_dev = [np.std(performance['acc'])]
    best_auc = [np.average(performance['auc'])]
    best_auc_std_dev = [np.std(performance['auc'])]
    for phase in range(14):
        feature_combo_accuracies = []
        feature_combo_auc = []
        names = []
        indexes = []
        acc_std_dev = []
        auc_std_dev = []
        for name, missing_label in labels.items():
            print('started loop with missing label ', missing_label)
            temp = np.ones(33, dtype=bool)
            temp[missing_label] = False
            try:
                temp[missing_labels] = False
            except:
                pass
            used_labels = full_list[temp]
            print('deep copying the database')
            train_state_data = deepcopy(database['train_state'])
            print('deep copying the validation base')
            validation_state_data = deepcopy(database['validation_state'])
            for episode in range(len(database['train_state'])):
                for tstep in range(len(database['train_state'][episode])):
                    train_state_data[episode][tstep] = [train_state_data[episode][tstep][used_label] for used_label in used_labels]
            for episode in range(len(database['validation_state'])):
                for tstep in range(len(database['validation_state'][episode])):
                    validation_state_data[episode][tstep] = [validation_state_data[episode][tstep][used_label] for used_label in used_labels]
            if test is not None:
                test_dataloader = deepcopy(test['validation_state'])
                for episode in range(len(test['validation_state'])):
                    for tstep in range(len(test['validation_state'][episode])):
                        test_dataloader[episode][tstep] = [test_dataloader[episode][tstep][used_label] for used_label in used_labels]
                reduced_test_dataset = RNNDataset(test_dataloader, test['validation_label'], test['test_pick_title'], args.batch_size, range_params=False)
            else:
                reduced_test_dataset=None
            reduced_train_dataset = RNNDataset(train_state_data, database['train_label'], database['train_pick_title'], args.batch_size, range_params=False)
            reduced_validation_dataset = RNNDataset(validation_state_data, database['validation_label'], database['validation_pick_title'], args.batch_size, range_params=False)
            
            args.input_dim = len(used_labels)
            print('using this many labels', len(used_labels))
            performance = {'auc': [], 'acc': []}
            for i in range(3):
                base_lstm = AppleClassifier(reduced_train_dataset,
                                            reduced_validation_dataset, vars(args),test_dataset=reduced_test_dataset)
                base_lstm.train()
                max_acc = base_lstm.get_best_performance(None)
                max_auc = np.max(base_lstm.AUC)
                performance['auc'].append(max_auc)
                performance['acc'].append(np.max(max_acc))
                print('model finished, saving now')
                base_lstm.save_data()
                print(performance)
            feature_combo_accuracies.append(np.average(performance['acc']))
            feature_combo_auc.append(np.average(performance['auc']))
            acc_std_dev.append(np.std(performance['acc']))
            auc_std_dev.append(np.std(performance['auc']))
            names.append(name)
            indexes.append(missing_label)
        best_one = np.argmax(feature_combo_auc)
        print('')
        print(f'best combination was {best_one}, with AUC {np.max(feature_combo_auc)} and Acc {np.max(feature_combo_auc)} and {np.shape(names)} inputs')
        print('')
        missing_names = missing_names + names[best_one]
        missing_labels.extend(indexes[best_one])
        best_accuracies.append(feature_combo_accuracies[best_one])
        best_acc_std_dev.append(acc_std_dev[best_one])
        best_auc.append(feature_combo_auc[best_one])
        best_auc_std_dev.append(auc_std_dev[best_one])
        worst_names.append(names[best_one])
        sizes.append(sizes[phase] - len(labels[names[best_one]]))
        labels.pop(names[best_one])
    print('ablation finished. best accuracies throughout were', best_accuracies)
    print('ablation finished. best AUC throughout were', best_auc)
    print('names removed in this order', worst_names)

    grasp_ablation_dict = {'num inputs': sizes, 'best accuracy': best_accuracies,
                           'accuracy std dev': best_acc_std_dev,'best auc': best_auc, 
                           'auc std dev': best_auc_std_dev,'names': worst_names}
    file = open('./generated_data/grasp_ablation_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M")
                + '.pkl', 'wb')
    pkl.dump(grasp_ablation_dict, file)
    file.close()
    return grasp_ablation_dict


def plot_ablation(ablation_dict):
    input_size = ablation_dict['num inputs']
    accuracy = ablation_dict['best accuracy']
    errs = ablation_dict['std dev']
    plt.errorbar(input_size,accuracy,yerr=errs)
    plt.show()

if __name__ == "__main__":            
    with open('./generated_data/grasp_ablation_data01_30_22_0122.pkl', 'rb') as file:
        pick_data = pkl.load(file)
    plot_ablation(pick_data)