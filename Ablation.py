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

def perform_ablation(database, args, validation=None):
    """
    Function to perform an ablation study to determine which feature is the
    most important for the classification
    @param train_dataset - a TensorDataset containing the training data
    @param test_dataset - a TensorDataset containing the testing data
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
    labels = {'Arm Force': [0, 1, 2], 
              'Arm Torque': [3,4,5],
              'IMU Acceleration': [6, 7, 8, 15, 16, 17, 24, 25, 26],
              'IMU Gyro': [9, 10, 11, 18, 19, 20, 27, 28, 29],
              'Finger Position': [12, 21, 30],
              'Finger Speed': [13, 22, 31],
              'Finger Effort': [14, 23, 32]}
    full_list = np.array(range(33))
    missing_labels = []
    missing_names = ''
    worst_names = []
    sizes = [33]
    full_train_loader = RNNDataset(database['train_state'], database['train_label'], args.batch_size)
    full_test_loader = RNNDataset(database['test_state'], database['test_label'], args.batch_size)
    if validation is not None:
        validation_loader = RNNDataset(validation['test_state'], validation['test_label'], args.batch_size)
    else:
        validation_loader = None
    performance = []
    for i in range(3):
        full_lstm = AppleClassifier(full_train_loader, full_test_loader, vars(args), validation_dataset=validation_loader)
        full_lstm.train()
        roc_max_acc = []
        for i in range(21):
            roc_acc, _, _ = full_lstm.evaluate(i*0.05, 'test')
            roc_max_acc.append(roc_acc)
        performance = np.max(roc_max_acc)
        print('model finished, saving now.')
#        print('performance: ', performance)
        full_lstm.save_data()
    best_accuracies = [np.average(performance)]
    best_acc_std_dev = [np.std(best_accuracies)]
    for phase in range(6):
        feature_combo_accuracies = []
        names = []
        indexes = []
        acc_std_dev = []
        for name, missing_label in labels.items():
            temp = np.ones(33, dtype=bool)
            temp[missing_label] = False
            try:
                temp[missing_labels] = False
            except:
                pass
            used_labels = full_list[temp]
            
            train_state_data = deepcopy(database['train_state'])
            test_state_data = deepcopy(database['test_state'])
            validation_dataloader = deepcopy(validation['test_state'])
            
            for episode in range(len(database['train_state'])):
                for tstep in range(len(database['train_state'][episode])):
                    train_state_data[episode][tstep] = [train_state_data[episode][tstep][used_label] for used_label in used_labels]
            for episode in range(len(database['test_state'])):
                for tstep in range(len(database['test_state'][episode])):
                    test_state_data[episode][tstep] = [test_state_data[episode][tstep][used_label] for used_label in used_labels]
            for episode in range(len(validation['test_state'])):
                for tstep in range(len(validation['test_state'][episode])):
                    validation_dataloader[episode][tstep] = [validation_dataloader[episode][tstep][used_label] for used_label in used_labels]

            reduced_train_dataset = RNNDataset(train_state_data, database['train_label'], args.batch_size)
            reduced_test_dataset = RNNDataset(test_state_data, database['test_label'], args.batch_size)
            reduced_validation_dataset = RNNDataset(validation_dataloader, validation['test_label'], args.batch_size)
            
            args.input_dim = len(used_labels)
            print('using this many labels', len(used_labels))
            performance = []
            for i in range(3):
                base_lstm = AppleClassifier(reduced_train_dataset,
                                            reduced_test_dataset, vars(args),validation_dataset=reduced_validation_dataset)
                base_lstm.train()
                roc_max_acc = []
                for i in range(21):
                    roc_acc, _, _ = base_lstm.evaluate(i*0.05,'test')
                    roc_max_acc.append(roc_acc)
#                print('roc max acc: ', roc_max_acc)
                performance.append(np.max(roc_max_acc))
                print('model finished, saving now')
#                print('performance: ', performance)
                base_lstm.save_data()
            feature_combo_accuracies.append(np.average(performance))
            acc_std_dev.append(np.std(performance))
            names.append(name)
            indexes.append(missing_label)
        best_one = np.argmax(feature_combo_accuracies)
        print('')
        print('best combination', best_one, np.max(feature_combo_accuracies), np.shape(names))
        print('')
#        print('lets just throw the data up here')
#        print('performance: ', performance)
#        print('feature combo accuracies: ', feature_combo_accuracies)
#        print('best name: ', names[best_one])
        missing_names = missing_names + names[best_one]
        missing_labels.extend(indexes[best_one])
        best_accuracies.append(feature_combo_accuracies[best_one])
        best_acc_std_dev.append(acc_std_dev[best_one])
        worst_names.append(names[best_one])
        sizes.append(sizes[phase] - len(labels[names[best_one]]))
        labels.pop(names[best_one])
#        print('best accuracies: ', best_accuracies)
    print('ablation finished. best accuracies throughout were', best_accuracies)
    print('names removed in this order', worst_names)

    grasp_ablation_dict = {'num inputs': sizes, 'best accuracy': best_accuracies,
                           'std dev': best_acc_std_dev, 'names': worst_names}
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