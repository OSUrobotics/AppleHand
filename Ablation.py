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
import json


class Trial():
    '''Class to hold data from individual trial'''

    def __init__(self, acc, auc):
        self.acc = acc
        self.auc = auc


class FeatureCombination():
    '''Class to hold n trials and generate average and std devs from them'''

    def __init__(self, labels, size):
        self.trials = []
        self.labels = labels
        self.size = size

    def add_trial(self, new_trial: Trial):
        self.trials.append(new_trial)

    def generate_means(self):
        acc_list = []
        auc_list = []
        for trial in self.trials:
            acc_list.append(trial.acc)
            auc_list.append(trial.auc)
        acc_mean = np.average(acc_list)
        acc_std_dev = np.std(acc_list)
        auc_mean = np.average(auc_list)
        auc_std_dev = np.std(auc_list)
        meta_dict = {'acc_mean': acc_mean, 'acc_std': acc_std_dev,
                     'acc_mean': auc_mean, 'acc_std': auc_std_dev,
                     'labels': self.labels, 'size': self.size}
        return meta_dict


class AblationLayer():
    '''Class to hold n feature combinations and find the best combination in that level'''

    def __init__(self, available_labels):
        self.label_list = available_labels
        self.combinations_raw_data = []
        self.combinations_generated_data = []

    def add_combination(self, new_combination: FeatureCombination):
        self.combinations_raw_data.append(new_combination)
        self.combinations_generated_data.append(new_combination.generate_means())

    def generate_bests(self):
        max_auc = -1
        for combo in self.combinations_generated_data:
            if combo['auc_mean'] > max_auc:
                max_auc = combo['auc_mean']
                max_acc = combo['acc_mean']
                acc_std_dev = combo['auc_std']
                auc_std_dev = combo['acc_std']
                used_labels = combo['labels']
                sizes = combo['size']
        worst_label = 'None'
        for label in self.label_list:
            if label not in used_labels:
                worst_label = label
                break

        best_dict = {'num inputs': sizes, 'best acc': max_acc,
                     'acc std dev': acc_std_dev, 'best auc': max_auc,
                     'auc std dev': auc_std_dev, 'worst label': worst_label}
        return best_dict


class AblationRunner():
    '''class to run a full ablation study'''

    def __init__(self, database, args):
        self.database = database
        self.args = args
        self.all_layers = []

    def build_dataset(self, used_labels, data_key):
        state_data = deepcopy(self.database[data_key + '_state'])
        for episode in range(len(self.database[data_key + '_state'])):
            for tstep in range(len(self.database[data_key + '_state'][episode])):
                state_data[episode][tstep] = [state_data[episode][tstep][used_label] for used_label in
                                              used_labels]
        reduced_dataset = RNNDataset(state_data, self.database[data_key + '_label'],
                                     self.database['train_pick_title'],
                                     self.args.batch_size, range_params=False)
        return reduced_dataset

    def generate_data_dict(self):
        data_dict = {'num inputs': [], 'best acc': [], 'acc std dev': [],
                     'best auc': [], 'auc std dev': [], 'worst label': []}
        for layer in self.all_layers:
            for key in layer.keys():
                data_dict[key].append(layer[key])
        return data_dict

    def perform_ablation(self, test=None):
        """
        Function to perform an ablation study to determine which feature is the
        most important for the classification
        @param train_dataset - a TensorDataset containing the training data
        @param validation_dataset - a TensorDataset containing the validationing data
        @param args - an argparse.Namespace containing hyperparameters for
        the classifiers
        @return - a dictionary containing the accuracy, standard deviation, number
        of inputs and removed features for every step of the ablation"""

        with open('./sensor_structures/sensor_structure.json') as sensor_file:
            labels = json.load(sensor_file)
        label_keys = [key for key in labels.keys()]
        self.all_layers.append(AblationLayer(label_keys))
        full_list = np.array(range(33))
        missing_labels = []
        max_size = 33
        full_train_loader = RNNDataset(self.database['train_state'], self.database['train_label'],
                                       self.database['train_pick_title'],
                                       self.args.batch_size, range_params=False)
        full_validation_loader = RNNDataset(self.database['validation_state'], self.database['validation_label'],
                                            self.database['validation_pick_title'], self.args.batch_size,
                                            range_params=False)
        if test is not None:
            test_loader = RNNDataset(self.database['test_state'], self.database['test_label'],
                                     self.database['test_pick_title'],
                                     self.args.batch_size, range_params=False)
        else:
            test_loader = None
        this_combo = FeatureCombination(label_keys, max_size)
        for i in range(3):
            full_lstm = AppleClassifier(full_train_loader, full_validation_loader, vars(self.args),
                                        test_dataset=test_loader)
            full_lstm.train()
            max_acc = full_lstm.get_best_performance(None)
            max_auc = np.max(full_lstm.AUC)
            ep_trial = Trial(max_acc, max_auc)
            this_combo.add_trial(ep_trial)
            print('model finished, saving now.')
            full_lstm.save_data()
        self.all_layers[-1].add_combination(this_combo)
        for phase in range(14):
            self.all_layers.append(AblationLayer(labels))
            for name, missing_label in labels.items():
                print('started loop with missing label ', name)
                used_label_list = [key for key in labels.keys()]
                used_label_list.remove(name)
                temp = np.ones(33, dtype=bool)
                temp[missing_label] = False
                try:
                    temp[missing_labels] = False
                except:
                    pass
                used_labels = full_list[temp]
                print('building train dataset')
                reduced_train_dataset = self.build_dataset(used_labels, 'train')
                print('building validation dataset')
                reduced_validation_dataset = self.build_dataset(used_labels, 'validation')
                if test is not None:
                    print('building test dataset')
                    reduced_test_dataset = self.build_dataset(used_labels, 'test')
                else:
                    reduced_test_dataset = None

                self.args.input_dim = len(used_labels)
                print('using this many labels', len(used_labels))
                this_combo = FeatureCombination(used_label_list, len(used_labels))
                for i in range(3):
                    base_lstm = AppleClassifier(reduced_train_dataset,
                                                reduced_validation_dataset, vars(self.args),
                                                test_dataset=reduced_test_dataset)
                    base_lstm.train()
                    max_acc = base_lstm.get_best_performance(None)
                    max_auc = np.max(base_lstm.AUC)
                    ep_trial = Trial(max_acc, max_auc)
                    this_combo.add_trial(ep_trial)
                    print('model finished, saving now')
                    base_lstm.save_data()
                self.all_layers[-1].add_combination(this_combo)
            layer_bests = self.all_layers[-1].generate_bests()
            print('')
            print(
                f"best combination had AUC {layer_bests['best auc']} and Acc {layer_bests['best acc']} and {layer_bests['num inputs']} inputs")
            print('')
            labels.pop(layer_bests['worst label'])

        ablation_dict = self.generate_data_dict()
        print('ablation finished. best accuracies throughout were', ablation_dict['best acc'])
        print('ablation finished. best AUC throughout were', ablation_dict['best auc'])
        print('names removed in this order', ablation_dict['worst label'])

        file = open('./generated_data/grasp_ablation_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M")
                    + '.pkl', 'wb')
        pkl.dump(ablation_dict, file)
        file.close()
        return ablation_dict

    def plot_ablation(self):
        ablation_dict = self.generate_data_dict()
        input_size = ablation_dict['num inputs']
        accuracy = ablation_dict['best acc']
        errs = ablation_dict['std dev']
        plt.errorbar(input_size, accuracy, yerr=errs)
        plt.show()


if __name__ == "__main__":
    print('nope, try calling from pickclassifier_main.py by running with arg --ablate=True')
