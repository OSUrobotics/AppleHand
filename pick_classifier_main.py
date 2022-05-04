#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:54:09 2021

@author: Nigel Swenson
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle as pkl
# from raw_csv_process import process_data
# from simple_csv_process import simple_process_data, process_data_iterable
from new_csv_process import GraspProcessor
import argparse
from AppleClassifier import AppleClassifier
from Ablation import perform_ablation
#from utils import RNNDataset
from alt_utils import RNNDataset
# from itertools import islice
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime
from copy import deepcopy


class ExperimentHandler:
    def __init__(self):
        self.setup_args()
        self.data_processor = GraspProcessor()
        self.test_dataset = []
        self.validation_dataset = []
        self.labels = {'Arm Force': [0, 1, 2],
                       'Arm Torque': [3, 4, 5],
                       'IMU Acceleration': [6, 7, 8, 15, 16, 17, 24, 25, 26],
                       'IMU Gyro': [9, 10, 11, 18, 19, 20, 27, 28, 29],
                       'Finger Position': [12, 21, 30],
                       'Finger Speed': [13, 22, 31],
                       'Finger Effort': [14, 23, 32]}
        if self.args.data_path is not None:
            if self.args.pick:
                self.data_processor.process_full_test(self.args.data_path)
            else:
                self.data_processor.process_data_iterable(self.args.data_path)
        if self.args.validation_path is not None:
            if self.args.pick:
                self.data_processor.process_full_test(self.args.validation_path, validate=True)
            else:
                self.data_processor.process_data_iterable(self.args.validation_path, validate=True)

        if self.args.pick:
            print('opening apple grasp and pick datset')
            file = open('combined_train_test_dataset.pkl', 'rb')
            self.test_data = pkl.load(file)
            self.test_data = self.make_float(self.test_data)
            file.close()
            if self.args.validate:
                file = open('combined_validation_dataset.pkl', 'rb')
                self.validation_data = pkl.load(file)
                self.validation_data = self.make_float(self.validation_data)
                file.close()
        else:
            file = open('train_test_grasp_dataset.pkl', 'rb')
            print('opening apple grasp datset')
            self.test_data = pkl.load(file)
            self.test_data = self.make_float(self.test_data)
            file.close()
            if self.args.validate:
                file = open('validation_grasp_dataset.pkl', 'rb')
                self.validation_data = pkl.load(file)
                self.validation_data = self.make_float(self.validation_data)
                file.close()

        self.test_dataset = []
        self.train_dataset = []
        self.build_dataset(False)
        if self.args.validate:
            self.validation_dataset = []
            self.build_dataset(True)

        self.classifiers = []
        self.data_dict = []
        self.figure_count = 1

    @staticmethod
    def make_data_name(model_name):
        ind = model_name.find('model')
        return model_name[0:ind] + 'data' + model_name[ind + 5:]

    @staticmethod
    def make_float(indict):
        for key in indict.keys():
            try:
                indict[key] = indict[key].astype(float)
            except AttributeError:
                pass
        return indict

    def build_dataset(self, validate=False):
        params = None
        if validate:
            params = np.load('proxy_mins_and_maxs.npy', allow_pickle=True)
            params = params.item()
            print('validation params',params)
            if self.args.used_features is None:
                self.validation_dataset = RNNDataset(self.validation_data['test_state'],
                                                     self.validation_data['test_label'],
                                                     self.validation_data['test_pick_title'], self.args.batch_size,
                                                     range_params=params)
            else:
                used_labels = []
                for label in self.args.used_features:
                    used_labels.extend(self.labels[label])
                self.validation_dataset = RNNDataset(
                    list(np.array(self.validation_data['test_state'])[:, :, used_labels]),
                    self.validation_data['test_label'], self.validation_data['test_pick_title'], self.args.batch_size,
                    range_params=params)
        else:
            print('building dataset')
            if self.args.used_features is None:
                print('first dataset')
                self.train_dataset = RNNDataset(self.test_data['train_state'], self.test_data['train_label'],
                                                self.test_data['train_pick_title'], self.args.batch_size, range_params=params)
                params = self.train_dataset.get_params()
                print('params from train dataset',params)
                print('second dataset')
                self.test_dataset = RNNDataset(self.test_data['test_state'], self.test_data['test_label'],
                                               self.test_data['test_pick_title'], self.args.batch_size, range_params=params)
                np.save('proxy_mins_and_maxs', params)
            else:
                used_labels = []
                for label in self.args.used_features:
                    used_labels.extend(self.labels[label])
                train_state_data = deepcopy(self.test_data['train_state'])
                test_state_data = deepcopy(self.test_data['test_state'])
                for episode in range(len(self.test_data['train_state'])):
                    for tstep in range(len(self.test_data['train_state'][episode])):
                        train_state_data[episode][tstep] = [train_state_data[episode][tstep][used_label] for used_label
                                                            in used_labels]
                for episode in range(len(self.test_data['test_state'])):
                    for tstep in range(len(self.test_data['test_state'][episode])):
                        test_state_data[episode][tstep] = [test_state_data[episode][tstep][used_label] for used_label in
                                                           used_labels]
                self.train_dataset = RNNDataset(train_state_data, self.test_data['train_label'],
                                                self.test_data['pick_title'], self.args.batch_size, range_params=params)
                params = self.train_dataset.get_params()
                self.test_dataset = RNNDataset(test_state_data, self.test_data['test_label'],
                                               self.test_data['pick_title'], self.args.batch_size, range_params=params)
            np.save('proxy_mins_and_maxs', params)

    def setup_args(self, args=None):
        """ Set important variables based on command line arguments OR passed on argument values
        returns: Full set of arguments to be parsed"""
        main_path = os.path.abspath(__file__)
        main_path = os.path.dirname(main_path)
        main_path = os.path.join(main_path, 'raw_data/')
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_type", default="LSTM", type=str)  # RNN used
        parser.add_argument("--epochs", default=100, type=int)  # num epochs trained for
        parser.add_argument("--layers", default=4, type=int)  # num layers in RNN
        parser.add_argument("--hiddens", default=100, type=int)  # num hidden nodes per layer
        parser.add_argument("--drop_prob", default=0.2, type=float)  # drop probability
        parser.add_argument("--reduced", default=True, type=bool)  # flag indicating to reduce data to only grasp part
        parser.add_argument("--data_path",
                            default=None,
                            type=str)  # path to unprocessed data
        parser.add_argument("--policy", default=None, type=str)  # filepath to trained policy
        parser.add_argument("--ablate", default=False, type=bool)  # flag to determine if ablation should be run
        parser.add_argument("--plot_acc", default=False,
                            type=bool)  # flag to determine if we want to plot the eval acc over epochs
        parser.add_argument("--plot_loss", default=False,
                            type=bool)  # flag to determine if we want to plot the loss over epochs
        parser.add_argument("--plot_ROC", default=False, type=bool)  # flag to determine if we want to plot an ROC
        parser.add_argument("--check_RF", default=False,
                            type=bool)  # flag to determine if we want to compare LSTM to RF on an ROC curve
        parser.add_argument("--plot_TP_FP", default=False,
                            type=bool)  # flag to determine if we want to plot the TP and FP over epochs
        parser.add_argument("--plot_example", default=False,
                            type=bool)  # flag to determine if we want to plot an episode and the networks performance
        parser.add_argument("--goal", default="grasp", type=str)  # desired output of the lstm
        parser.add_argument("--compare_policy", default=False,
                            type=bool)  # bool to train new policy and compare to old policy in all plots
        parser.add_argument("--batch_size", default=1, type=int)  # number of episodes in a batch during training
        parser.add_argument("--used_features", default=None, type=str)
        parser.add_argument("--validate", default=False, type=bool)
        parser.add_argument("--pick", default=False, type=bool)
        parser.add_argument("--validation_path", default=None, type=str)
        parser.add_argument("--s_f_bal", default=None, type=float)

        args = parser.parse_args()
        if args.used_features is not None:
            args.used_features = args.used_features.split(',')
        self.args = args
        return args

    def run_experiment(self):
        # Load policy if it exists, if not train a new one

        if self.args.policy is None or self.args.compare_policy:
            try:
                classifier = AppleClassifier(self.train_dataset, self.test_dataset, vars(self.args),
                                             validation_dataset=self.validation_dataset)
            except AttributeError:
                classifier = AppleClassifier(self.train_dataset, self.test_dataset, vars(self.args))
#            input('stahp')
            classifier.train()
            print('model finished, saving now')
            classifier.save_data()
            classifier.save_model()
        else:
            classifier = AppleClassifier(self.train_dataset, self.test_dataset, vars(self.args))
            classifier.load_model(self.args.policy)
            classifier.load_model_data(self.args.policy)
        self.classifiers.append(classifier)
        self.data_dict.append(classifier.get_data_dict())
        print(self.data_dict[0])
        if self.args.compare_policy:
            old_classifier = AppleClassifier(self.train_dataset, self.test_dataset, vars(self.args))
            old_classifier.load_model(self.args.policy)
            old_classifier.load_model_data(self.args.policy)
            self.classifiers.append(old_classifier)
            self.data_dict.append(old_classifier.get_data_dict())
        self.figure_count = 1

        # Plot accuracy over time if desired
        if self.args.plot_acc:
            self.plot_acc()

        # Plot loss over time if desired
        if self.args.plot_loss:
            self.plot_loss()

        # Plot TP and FP rate over time if desired
        if self.args.plot_TP_FP:
            self.plot_TP_FP()

        # Train a random forset on the last datapoint in the series if desired
        if self.args.check_RF:
            self.train_RF()

        # Plot an ROC if desired
        if self.args.plot_ROC:
            self.plot_ROC()

        # Find validation ROC if desired
        if self.args.validate:
            self.validation()

        # Visualize all plots made
        plt.show()

        # Plot single example if desired
        if self.args.plot_example:
            self.plot_example()

        # Perform ablation on feature groups if desired
        if self.args.ablate:
            perform_ablation(self.test_data, self.args, None)

    def plot_acc(self):
        print('Plotting classifier accuracy over time')
        legend = []
        acc_plot = plt.figure(self.figure_count)
        for data in self.data_dict:
            plt.plot(data['steps'], [max(accs) for accs in data['acc']])
#            plt.plot(data['steps'], [max(accs) for accs in data['train_acc']])
            try:
                plt.plot(data['steps'], [max(accs) for accs in data['validation_acc']])
            except:
                print('no validation acc')
                pass
            legend.append(data['ID'] + ' accuracy')
#            legend.append(data['ID'] + ' training accuracy')
            legend.append(data['ID'] + ' validation accuracy')
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        self.figure_count += 1

    def plot_loss(self):
        legend = []
        print('Plotting classifier loss over time')
        loss_plot = plt.figure(self.figure_count)
        for data in self.data_dict:
            plt.plot(data['steps'][1:], data['loss'][1:])
            legend.append(data['ID'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        self.figure_count += 1

    def plot_TP_FP(self):
        legend = []
        print('Plotting true positive and false positive rate over time')
        TPFP_plot = plt.figure(self.figure_count)
        for policy in self.classifiers:
            _, best_TP, best_FP, _ = policy.get_best_performance([0, -1])
            ID = policy.identifier
            plt.plot(range(len(best_TP)), best_TP)
            plt.plot(range(len(best_FP)), best_FP)
            legend.append('TP_' + ID)
            legend.append('FP_' + ID)
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('True and False Positive Rate')
        plt.title('True and False Positive Rate')
        self.figure_count += 1

    def plot_ROC(self):
        legend = []
        count = 0
        if self.args.check_RF:
            RF_train_data = []
            train_label = []
            for episode, episode_label in zip(self.test_data['train_state'], self.test_data['train_label']):
                RF_train_data.append(episode[-1])
                train_label.append(episode_label[-1])
            test_data = []
            test_label = []
            for episode, episode_label in zip(self.test_data['test_state'], self.test_data['test_label']):
                test_data.append(episode[-1])
                test_label.append(episode_label[-1])
            RF = RandomForestClassifier(n_estimators=100, random_state=42)
            RF.fit(RF_train_data, train_label)
            test_results = RF.predict(test_data)
            print("RF classifier accuracy:", metrics.accuracy_score(test_label, test_results))
            svc_disp = metrics.RocCurveDisplay.from_estimator(RF, test_data, test_label)
            legend.append('RF classifier')
        else:
            pass
        ROC_plot = plt.figure(self.figure_count)
        for policy in self.classifiers:
            max_acc, best_TP, best_FP, _ = policy.get_best_performance([0, -1])
            print('best test accuracy is ', max(max_acc))
            best_epoch = np.argmax(max_acc)
            plt.plot(policy.FP_rate[best_epoch], policy.TP_rate[best_epoch])
            legend.append(policy.identifier)
            count += 1

        baseline = [0, 1]
        legend.append('Random Baseline')
        plt.plot(baseline, baseline, linestyle='--')
        plt.legend(legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC')

        self.figure_count += 1

    def train_RF(self):
        RF_train_data = []
        train_label = []
        for episode, episode_label in zip(self.test_data['train_state'], self.test_data['train_label']):
            RF_train_data.append(episode[0])
            train_label.append(episode_label[-1])
        test_data = []
        test_label = []
        for episode, episode_label in zip(self.test_data['test_state'], self.test_data['test_label']):
            test_data.append(episode[0])
            test_label.append(episode_label[-1])
        RF = RandomForestClassifier(n_estimators=100, random_state=42)
        RF.fit(RF_train_data, train_label)
        test_results = RF.predict(test_data)
        print("RF classifier accuracy:", metrics.accuracy_score(test_label, test_results))
        np.save('./models/RF_models/RF_classifier_' + datetime.datetime.now().strftime("%m_%d_%y_%H%M"),
                np.array(metrics.accuracy_score(test_label, test_results)))

    def validation(self):
        legend = []
        print('Plotting an ROC curve')
        count = 0
        if self.args.check_RF:
            RF_train_data = []
            train_label = []
            for episode, episode_label in zip(self.test_data['train_state'], self.test_data['train_label']):
                RF_train_data.append(episode[-1])
                train_label.append(episode_label[-1])
            test_data = []
            test_label = []
            for episode, episode_label in zip(self.validation_data['test_state'], self.validation_data['test_label']):
                test_data.append(episode[-1])
                test_label.append(episode_label[-1])
            RF = RandomForestClassifier(n_estimators=100, random_state=42)
            RF.fit(RF_train_data, train_label)
            test_results = RF.predict(test_data)
            print("RF classifier accuracy:", metrics.accuracy_score(test_label, test_results))
            svc_disp = metrics.RocCurveDisplay.from_estimator(RF, test_data, test_label)
            legend.append('RF classifier')
        else:
            pass
        ROC_plot = plt.figure(self.figure_count)
        flag = False
        for policy in self.classifiers:
            policy.load_dataset(self.train_dataset, self.validation_dataset)
            try:
                acc, TP, FP, AUC = policy.evaluate(threshold=0.5, current=flag, test_set='validation')
            except RuntimeError:
                acc, TP, FP, ACU = policy.evaluate(threshold=0.5, current=True, test_set='validation')
            plt.plot(FP, TP)
            legend.append(policy.identifier)
            count += 1
            best_thold = np.argmax(acc)
            print('best validation accuracy is ', acc[best_thold], TP[best_thold], FP[best_thold])
        baseline = [0, 1]
        legend.append('Random Baseline')
        plt.plot(baseline, baseline, linestyle='--')
        plt.legend(legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC')

        self.figure_count += 1

    def plot_example(self):

        legend = ['Correct Output']
        print('Plotting single episode')
        count = 0
        single_episode_flag = True
        while single_episode_flag:
            plt.clf()
            single_episode_plot = plt.figure(self.figure_count)
            plot_bounds = [self.classifiers[0].visualization_range[0], self.classifiers[0].visualization_range[1]]
            first_plot = True
            for policy in self.classifiers:
                x, y, outputs, e_name = policy.evaluate_episode()
                if first_plot:
                    plt.scatter(
                        range(len(y)), y,
                        c='green')
                    first_plot = False
                    legend.append(policy.identifier + ' raw output')
                    legend.append(policy.identifier + ' rounded output')
                count += 1
                rounded_outputs = []
                for i in range(len(outputs)):
                    if outputs[i] >= 0.75:
                        rounded_outputs.append(1.05)
                    elif outputs[i] <= 0.25:
                        rounded_outputs.append(0.05)
                    else:
                        rounded_outputs.append(0.5)
                plt.scatter(range(len(outputs)), outputs, marker="+")
                plt.scatter(range(len(rounded_outputs)), rounded_outputs, marker="x")

            plt.legend(legend)

            plt.ylim(-0.05, 1.1)
            print('just did e_name', e_name[0][0], ' last thing was ', y[-1][0])
            plt.show()
            self.figure_count += 1

            flag_choice = input('generate another plot? y/n')
            if flag_choice.lower() == 'n':
                single_episode_flag = False
            count += 1
        return e_name, outputs, y


if __name__ == "__main__":
    # Read in arguments from command line
    experiments = ExperimentHandler()
    experiments.run_experiment()
