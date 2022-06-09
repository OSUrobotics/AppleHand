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
import time
# from raw_csv_process import process_data
# from simple_csv_process import simple_process_data, process_data_iterable
from csv_process import GraspProcessor
import argparse
from AppleClassifier import AppleClassifier
from Ablation import perform_ablation
from utils import RNNDataset
# from itertools import islice
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime
from copy import deepcopy


class ExperimentHandler:
    def __init__(self,alt_args=None):
        self.setup_args()
        print(self.args.phase)
        if alt_args is not None:
            self.modify_args(alt_args)
        print(self.args.phase)
        self.data_processor = GraspProcessor()
        self.validation_dataset = []
        self.test_dataset = []
        self.labels = {'Arm Force': [0, 1, 2],
                       'Arm Torque': [3, 4, 5],
                       'IMU Acceleration': [6, 7, 8, 15, 16, 17, 24, 25, 26],
                       'IMU Gyro': [9, 10, 11, 18, 19, 20, 27, 28, 29],
                       'Finger Position': [12, 21, 30],
                       'Finger Speed': [13, 22, 31],
                       'Finger Effort': [14, 23, 32]}
        if self.args.data_path is not None:
            if self.args.phase.lower() == 'full':
                self.data_processor.process_full_validation(self.args.data_path)
            elif self.args.phase.lower() == 'grasp':
                self.data_processor.process_data_iterable(self.args.data_path)
            elif self.args.phase.lower() == 'pick':
                self.data_processor.process_data_iterable(self.args.data_path)
        if self.args.test_path is not None:
            if self.args.phase.lower() == 'full':
                self.data_processor.process_full_validation(self.args.test_path, evaluate=True)
            elif self.args.phase.lower() == 'grasp':
                self.data_processor.process_data_iterable(self.args.test_path, evaluate=True)
            elif self.args.phase.lower() == 'pick':
                self.data_processor.process_data_iterable(self.args.test_path, evaluate=True)

        if self.args.phase.lower() == 'full':
            print('opening full apple datset')
            file = open('./datasets/combined_train_validation_dataset.pkl', 'rb')
            self.validation_data = pkl.load(file)
            self.validation_data = self.make_float(self.validation_data)
            file.close()
            if self.args.evaluate:
                file = open('./datasets/combined_test_dataset.pkl', 'rb')
                self.test_data = pkl.load(file)
                self.test_data = self.make_float(self.test_data)
                file.close()
        elif self.args.phase.lower() == 'grasp':
            file = open('./datasets/train_validation_grasp_dataset.pkl', 'rb')
            print('opening apple grasp datset')
            self.validation_data = pkl.load(file)
            self.validation_data = self.make_float(self.validation_data)
            file.close()
            if self.args.evaluate:
                file = open('./datasets/test_grasp_dataset.pkl', 'rb')
                self.test_data = pkl.load(file)
                self.test_data = self.make_float(self.test_data)
                file.close()
        elif self.args.phase.lower() == 'pick':
            file = open('./datasets/train_validation_pick_dataset.pkl', 'rb')
            print('opening apple pick datset')
            self.validation_data = pkl.load(file)
            self.validation_data = self.make_float(self.validation_data)
            file.close()
            if self.args.evaluate:
                file = open('./datasets/test_pick_dataset.pkl', 'rb')
                self.test_data = pkl.load(file)
                self.test_data = self.make_float(self.test_data)
                file.close()

        self.validation_dataset = []
        self.train_dataset = []
        self.build_dataset(False)
        if self.args.evaluate:
            self.test_dataset = []
            self.build_dataset(True)

        self.classifiers = []
        self.data_dict = []
        self.figure_count = 1

    def modify_args(self,alt_args):
        self.args.phase = alt_args

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

    def build_dataset(self, evaluate=False):
        params = None
        if evaluate:
            params = np.load('proxy_mins_and_maxs.npy', allow_pickle=True)
            params = params.item()
            print('test params',params)
            if self.args.used_features is None:
                self.test_dataset = RNNDataset(self.test_data['validation_state'],
                                                     self.test_data['validation_label'],
                                                     self.test_data['validation_pick_title'], self.args.batch_size,
                                                     range_params=False)
            else:
                used_labels = []
                for label in self.args.used_features:
                    used_labels.extend(self.labels[label])
                self.test_dataset = RNNDataset(
                    list(np.array(self.test_data['validation_state'])[:, :, used_labels]),
                    self.test_data['validation_label'], self.test_data['validation_pick_title'], self.args.batch_size,
                    range_params=False)
        else:
            print('building dataset')
            if self.args.used_features is None:
                print('first dataset')
                self.train_dataset = RNNDataset(self.validation_data['train_state'], self.validation_data['train_label'],
                                                self.validation_data['train_pick_title'], self.args.batch_size, range_params=False)
                params = self.train_dataset.get_params()
                print('params from train dataset',params)
                print('second dataset')
                self.validation_dataset = RNNDataset(self.validation_data['validation_state'], self.validation_data['validation_label'],
                                               self.validation_data['validation_pick_title'], self.args.batch_size, range_params=False)
                np.save('proxy_mins_and_maxs', params)
            else:
                used_labels = []
                for label in self.args.used_features:
                    used_labels.extend(self.labels[label])
                train_state_data = deepcopy(self.validation_data['train_state'])
                validation_state_data = deepcopy(self.validation_data['validation_state'])
                for episode in range(len(self.validation_data['train_state'])):
                    for tstep in range(len(self.validation_data['train_state'][episode])):
                        train_state_data[episode][tstep] = [train_state_data[episode][tstep][used_label] for used_label
                                                            in used_labels]
                for episode in range(len(self.validation_data['validation_state'])):
                    for tstep in range(len(self.validation_data['validation_state'][episode])):
                        validation_state_data[episode][tstep] = [validation_state_data[episode][tstep][used_label] for used_label in
                                                           used_labels]
                self.train_dataset = RNNDataset(train_state_data, self.validation_data['train_label'],
                                                self.validation_data['pick_title'], self.args.batch_size, range_params=False)
                params = self.train_dataset.get_params()
                self.validation_dataset = RNNDataset(validation_state_data, self.validation_data['validation_label'],
                                               self.validation_data['pick_title'], self.args.batch_size, range_params=False)
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
        parser.add_argument("--plot_AUC", default=False,
                            type=bool)  # flag to determine if we want to plot the TP and FP over epochs
        parser.add_argument("--plot_example", default=False,
                            type=bool)  # flag to determine if we want to plot an episode and the networks performance
        parser.add_argument("--goal", default="grasp", type=str)  # desired output of the lstm
        parser.add_argument("--compare_policy", default=False,
                            type=bool)  # bool to train new policy and compare to old policy in all plots
        parser.add_argument("--batch_size", default=1, type=int)  # number of episodes in a batch during training
        parser.add_argument("--used_features", default=None, type=str)
        parser.add_argument("--evaluate", default=False, type=bool)
        parser.add_argument("--phase", default='full', type=str)
        parser.add_argument("--test_path", default=None, type=str)
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
                classifier = AppleClassifier(self.train_dataset, self.validation_dataset, vars(self.args),
                                             test_dataset=self.test_dataset)
            except AttributeError:
                print('THERE WAS AN ATTRIBUTE ERROR!')
                classifier = AppleClassifier(self.train_dataset, self.validation_dataset, vars(self.args))
#            input('stahp')
            if self.args.ablate:
                perform_ablation(self.validation_data, self.args, None)
            else:
                classifier.train()
                classifier.save_metadata()
                print('model finished, saving now')
                classifier.save_data()
                classifier.save_model()
        else:
            classifier = AppleClassifier(self.train_dataset, self.validation_dataset, vars(self.args))
            classifier.load_model(self.args.policy)
            classifier.load_model_data(self.args.policy)
        self.classifiers.append(classifier)
        self.data_dict.append(classifier.get_data_dict())
        # print(self.data_dict[0])
        if self.args.compare_policy:
            old_classifier = AppleClassifier(self.train_dataset, self.validation_dataset, vars(self.args))
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
        if self.args.plot_AUC:
            self.plot_AUC()

        # Train a random forset on the last datapoint in the series if desired
        if self.args.check_RF:
            self.train_RF()

        # Plot an ROC if desired
        if self.args.plot_ROC:
            self.plot_ROC()

        # Find test ROC if desired
        if self.args.evaluate:
            self.test()

        # Visualize all plots made
#        plt.show()

        # Plot single example if desired
        if self.args.plot_example:
            self.plot_example()

    def plot_acc(self):
        print('Plotting classifier accuracy over time')
        legend = []
        acc_plot = plt.figure(self.figure_count)
        for data in self.data_dict:
            plt.plot(data['steps'], [max(accs) for accs in data['acc']])
#            plt.plot(data['steps'], [max(accs) for accs in data['train_acc']])
            try:
                plt.plot(data['steps'], [max(accs) for accs in data['test_acc']])
            except:
                print('no test acc')
                pass
            legend.append(data['ID'] + ' accuracy')
#            legend.append(data['ID'] + ' training accuracy')
            legend.append(data['ID'] + ' test accuracy')
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
            try:
                plt.plot(data['steps'][1:], data['loss'])
            except:
                plt.plot(data['steps'][1:], data['loss'][1:])
            legend.append(data['ID'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        self.figure_count += 1

    def plot_AUC(self):
        print('Plotting classifier AUC over time')
        legend = []
        AUC_plot = plt.figure(self.figure_count)
        for data in self.data_dict:
            plt.plot(data['steps'], data['AUC'])
            try:
                plt.plot(data['steps'],data['test_AUC'])
            except:
                print('no test acc')
                pass
            legend.append(data['ID'] + ' AUC')
            #            legend.append(data['ID'] + ' training accuracy')
            legend.append(data['ID'] + ' test AUC')
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('AUC')
        plt.title('AUC Curve')
        self.figure_count += 1

    def plot_ROC(self):
        legend = []
        count = 0
        if self.args.check_RF:
            RF_train_data = []
            train_label = []
            for episode, episode_label in zip(self.validation_data['train_state'], self.validation_data['train_label']):
                RF_train_data.append(episode[-1])
                train_label.append(episode_label[-1])
            validation_data = []
            validation_label = []
            for episode, episode_label in zip(self.validation_data['validation_state'], self.validation_data['validation_label']):
                validation_data.append(episode[-1])
                validation_label.append(episode_label[-1])
            RF = RandomForestClassifier(n_estimators=100, random_state=42)
            RF.fit(RF_train_data, train_label)
            validation_results = RF.predict(validation_data)
            print("RF classifier accuracy:", metrics.accuracy_score(validation_label, validation_results))
            svc_disp = metrics.RocCurveDisplay.from_estimator(RF, validation_data, validation_label)
            legend.append('RF classifier')
        else:
            pass
        ROC_plot = plt.figure(self.figure_count)
        for policy in self.classifiers:
            max_acc, best_TP, best_FP, _ = policy.get_best_performance([0, -1])
            print('best validation accuracy is ', max(max_acc))
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
        plt.title('validation ROC')

        self.figure_count += 1

    def train_RF(self):
        RF_train_data = []
        train_label = []
        for episode, episode_label in zip(self.validation_data['train_state'], self.validation_data['train_label']):
            RF_train_data.append(episode[0])
            train_label.append(episode_label[-1])
        validation_data = []
        validation_label = []
        for episode, episode_label in zip(self.validation_data['validation_state'], self.validation_data['validation_label']):
            validation_data.append(episode[0])
            validation_label.append(episode_label[-1])
        RF = RandomForestClassifier(n_estimators=100, random_state=42)
        RF.fit(RF_train_data, train_label)
        validation_results = RF.predict(validation_data)
        print("RF classifier accuracy:", metrics.accuracy_score(validation_label, validation_results))
        np.save('./models/RF_models/RF_classifier_' + datetime.datetime.now().strftime("%m_%d_%y_%H%M"),
                np.array(metrics.accuracy_score(validation_label, validation_results)))

    def test(self):
        legend = []
        print('Plotting an ROC curve')
        count = 0
        if self.args.check_RF:
            RF_train_data = []
            train_label = []
            for episode, episode_label in zip(self.validation_data['train_state'], self.validation_data['train_label']):
                RF_train_data.append(episode[-1])
                train_label.append(episode_label[-1])
            validation_data = []
            validation_label = []
            for episode, episode_label in zip(self.test_data['validation_state'], self.test_data['validation_label']):
                validation_data.append(episode[-1])
                validation_label.append(episode_label[-1])
            RF = RandomForestClassifier(n_estimators=100, random_state=42)
            RF.fit(RF_train_data, train_label)
            validation_results = RF.predict(validation_data)
            print("RF classifier accuracy:", metrics.accuracy_score(validation_label, validation_results))
            svc_disp = metrics.RocCurveDisplay.from_estimator(RF, validation_data, validation_label)
            legend.append('RF classifier')
        else:
            pass
        ROC_plot = plt.figure(self.figure_count)
        flag = False
        for policy in self.classifiers:
            policy.load_dataset(self.train_dataset, self.test_dataset)
            try:
                TP, FP, TP_group, FP_group, acc, acc_group = policy.evaluate(threshold=0.5, current=flag, test_set='test', ROC=True)
            except RuntimeError:
                TP, FP, TP_group, FP_group, acc, acc_group = policy.evaluate(threshold=0.5, current=True, test_set='test', ROC=True)
            plt.plot(FP, TP)
            # plt.plot(FP_group, TP_group)
            legend.append('policy')
            # legend.append('grouped policy')
            count += 1
            best_thold = np.argmax(acc)
            print('best test accuracy is ', acc[best_thold], TP[best_thold], FP[best_thold])
            best_group_thold = np.argmax(acc_group)
            print('best grouped test accuracy is ', acc_group[best_group_thold], TP_group[best_group_thold], FP_group[best_group_thold])
        baseline = [0, 1]
        legend.append('Random Baseline')
        plt.plot(baseline, baseline, linestyle='--')
        plt.legend(legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('test ROC')

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
                    plt.scatter(range(len(y[0])), y[0], c='green')
                    first_plot = False
                    legend.append(policy.identifier + ' raw output')
                    legend.append(policy.identifier + ' rounded output')
                count += 1
                rounded_outputs = []
                for i in range(len(outputs[0])):
                    if outputs[0][i] >= 0.75:
                        rounded_outputs.append(1.05)
                    elif outputs[0][i] <= 0.25:
                        rounded_outputs.append(0.05)
                    else:
                        rounded_outputs.append(0.5)
                plt.scatter(range(len(outputs[0])), outputs[0], marker="+")
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

def run_experiment_group():
    num_trials = 3
    things_to_run = ['grasp', 'pick', 'full']
    start_data_path = './raw_data/RAL22_Paper/'
    proxy = '4_proxy_winter22_x5'
    real = '6_real_fall21_x5'
    grasp = '/GRASP'
    pick = '/PICK'
    for i in num_trials:
        data_path=start_data_path+proxy
        pick=True

if __name__ == "__main__":
    # Read in arguments from command line
#    print('training 4 times with same params')
#    experiments = ExperimentHandler()
#    experiments.run_experiment()
    phases = ['full','grasp', 'pick']
    print('time to do some runtime analysis')
    times = []
    for j in range(3):
        print(f'starting {phases[j]} phase')
        for i in range(4):
            print(f'starting trial number {i}')
            start = time.time()
            experiments = ExperimentHandler(phases[j])
            experiments.run_experiment()
            end = time.time()
            print('training time = ',end - start)
            times.append(end-start)
    print(times)
    
#    
#    experiments = ExperimentHandler()
#    experiments.run_experiment()

