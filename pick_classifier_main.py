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
from raw_csv_process import process_data
from simple_csv_process import simple_process_data, process_data_iterable
import argparse
from AppleClassifier import AppleClassifier
from Ablation import perform_ablation
from utils import RNNDataset
from itertools import islice
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime


def make_data_name(model_name):
    ind = model_name.find('model')
    return model_name[0:ind] + 'data' + model_name[ind + 5:]


def build_dataset(database, args, validate=False):
    print(args.used_features)
    params = None
    if args.used_features is None:
        if validate:
            params = np.load('proxy_mins_and_maxs.npy', allow_pickle=True)
            params = params.item()
        train_dataset = RNNDataset(database['train_state'], database['train_label'], args.batch_size,range_params=params)
        params = train_dataset.get_params()
        test_dataset = RNNDataset(database['test_state'], database['test_label'], args.batch_size, range_params=params)
        np.save('proxy_mins_and_maxs', params)
    else:
        labels = {'Arm Force': [0, 1, 2],
                  'Arm Torque': [3, 4, 5],
                  'IMU Acceleration': [6, 7, 8, 15, 16, 17, 24, 25, 26],
                  'IMU Gyro': [9, 10, 11, 18, 19, 20, 27, 28, 29],
                  'Finger Position': [12, 21, 30],
                  'Finger Speed': [13, 22, 31],
                  'Finger Effort': [14, 23, 32]}
        used_labels = []
        for label in args.used_features:
            used_labels.extend(labels[label])
        if validate:
            params = np.load('proxy_mins_and_maxs.npy', allow_pickle=True)
            params = params.item()
        try:
            train_dataset = RNNDataset(list(np.array(database['train_state'])[:, :, used_labels]), database['train_label'], args.batch_size, range_params=params)
            params = train_dataset.get_params()
        except IndexError:
            train_dataset = None
        
        test_dataset = RNNDataset(list(np.array(database['test_state'])[:, :, used_labels]), database['test_label'], args.batch_size, range_params=params)
        np.save('proxy_mins_and_maxs',params)
    return train_dataset, test_dataset


def setup_args(args=None):
    """ Set important variables based on command line arguments OR passed on argument values
    returns: Full set of arguments to be parsed"""
    main_path = os.path.abspath(__file__)
    main_path = os.path.dirname(main_path)
    main_path = os.path.join(main_path, 'raw_data/')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="LSTM", type=str)  # RNN used
    parser.add_argument("--epochs", default=25, type=int)  # num epochs trained for
    parser.add_argument("--layers", default=4, type=int)  # num layers in RNN
    parser.add_argument("--hiddens", default=100, type=int)  # num hidden nodes per layer
    parser.add_argument("--drop_prob", default=0.2, type=float)  # drop probability
    parser.add_argument("--reduced", default=True, type=bool)  # flag indicating to reduce data to only grasp part
    parser.add_argument("--data_path",
                        default=main_path,
                        type=str)  # path to unprocessed data
    parser.add_argument("--policy", default=None, type=str)  # filepath to trained policy
    parser.add_argument("--ablate", default=False, type=bool)  # flag to determine if ablation should be run
    parser.add_argument("--plot_acc", default=False,
                        type=bool)  # flag to determine if we want to plot the eval acc over epochs
    parser.add_argument("--plot_loss", default=False,
                        type=bool)  # flag to determine if we want to plot the loss over epochs
    parser.add_argument("--plot_ROC", default=False, type=bool)  # flag to determine if we want to plot an ROC
    parser.add_argument("--check_RF", default=False, type=bool)  # flag to determine if we want to compare LSTM to RF on an ROC curve
    parser.add_argument("--plot_TP_FP", default=False,
                        type=bool)  # flag to determine if we want to plot the TP and FP over epochs
    parser.add_argument("--plot_example", default=False,
                        type=bool)  # flag to determine if we want to plot an episode and the networks perfromance
    parser.add_argument("--goal", default="grasp", type=str)  # desired output of the lstm
    parser.add_argument("--reprocess", default=False, type=bool)  # bool to manually reprocess data if changed
    parser.add_argument("--compare_policy", default=False,
                        type=bool)  # bool to train new policy and compare to old policy in all plots
    parser.add_argument("--batch_size", default=1, type=int)  # number of episodes in a batch during training
    parser.add_argument("--used_features", default=None, type=str)
    parser.add_argument("--validate", default=False, type=bool)
    parser.add_argument("--validation_path", default=None, type=str)

    args = parser.parse_args()
    if args.used_features is not None:
        args.used_features = args.used_features.split(',')
    return args


if __name__ == "__main__":
    # Read in arguments from command line
    args = setup_args()

    # Load processed data if it exists and process csvs if it doesn't
    if args.validation_path is not None:
        try:
            process_data_iterable(args.validation_path, validation=True)
        except:
            process_data(args.validation_path, validation=True) #right now this won't work
    if args.reprocess:
        try:
            process_data_iterable(args.data_path)
            file = open('apple_dataset.pkl', 'rb')
            pick_data = pkl.load(file)
            file.close()
        except:
            process_data(args.data_path)
            file = open('apple_dataset.pkl', 'rb')
            pick_data = pkl.load(file)
            file.close()
    else:
        try:
            file = open('apple_dataset.pkl', 'rb')
            pick_data = pkl.load(file)
            file.close()
        except FileNotFoundError:
            try:
                process_data_iterable(args.data_path)
                file = open('apple_dataset.pkl', 'rb')
                pick_data = pkl.load(file)
                file.close()
            except:
                process_data(args.data_path)
                file = open('apple_dataset.pkl', 'rb')
                pick_data = pkl.load(file)
                file.close()

    train_data, test_data = build_dataset(pick_data, args)
    if args.validate:
        try:
            file = open('validation_dataset.pkl', 'rb')
            validation_dataset = pkl.load(file)
            file.close()
            print(np.array(validation_dataset['test_state'])[:, :, 1])
            _, validation_data = build_dataset(validation_dataset, args, validate=True)
        except FileNotFoundError:
            print('No validation dataset, make sure you pass in the path to the validation data before running with validate=True')
            raise FileNotFoundError
    # Load policy if it exists, if not train a new one
    loaded_classifiers = []
    loaded_dicts = []
    if args.policy is None or args.compare_policy:
        classifier = AppleClassifier(train_data, test_data, vars(args))
        classifier.train()
        print('model finished, saving now')
        classifier.save_data()
        classifier.save_model()
    else:
        classifier = AppleClassifier(train_data, test_data, vars(args))
        classifier.load_model(args.policy)
        classifier.load_model_data(args.policy)
    loaded_classifiers.append(classifier)
    loaded_dicts.append(classifier.get_data_dict())
    if args.compare_policy:
        old_classifier = AppleClassifier(train_data, test_data, vars(args))
        old_classifier.load_model(args.policy)
        old_classifier.load_model_data(args.policy)
        loaded_classifiers.append(old_classifier)
        loaded_dicts.append(old_classifier.get_data_dict())
    figure_count = 1

    # Plot accuracy over time if desired
    if args.plot_acc:
        print('Plotting classifier accuracy over time')
        legend = []
        acc_plot = plt.figure(figure_count)
        for data in loaded_dicts:
            plt.plot(data['steps'], data['acc'])
            plt.plot(data['steps'], data['train_acc'])
            legend.append(data['ID'] + 'accuracy')
            legend.append(data['ID'] + 'training accuracy')
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        figure_count += 1

    # Plot loss over time if desired
    if args.plot_loss:
        legend = []
        print('Plotting classifier loss over time')
        loss_plot = plt.figure(figure_count)
        for data in loaded_dicts:
            plt.plot(data['steps'][1:], data['loss'][1:])
            legend.append(data['ID'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        figure_count += 1

    # Plot TP and FP rate over time if desired
    if args.plot_TP_FP:
        legend = []
        print('Plotting true positive and false positive rate over time')
        TPFP_plot = plt.figure(figure_count)
        for data in loaded_dicts:
            plt.plot(data['steps'], data['TP'])
            plt.plot(data['steps'], data['FP'])
            legend.append('TP_' + data['ID'])
            legend.append('FP_' + data['ID'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('True and False Positive Rate')
        plt.title('True and False Positive Rate')
        figure_count += 1

    # Train a random forset on the last datapoint in the series if desired
    if args.check_RF:
        RF_train_data = []
        train_label = []
        for episode, episode_label in zip(pick_data['train_state'], pick_data['train_label']):
            RF_train_data.append(episode[-1])
            train_label.append(episode_label[-1])
        test_data = []
        test_label = []
        for episode, episode_label in zip(pick_data['test_state'], pick_data['test_label']):
            test_data.append(episode[-1])
            test_label.append(episode_label[-1])
        RF = RandomForestClassifier(n_estimators=100, random_state=42)
        RF.fit(RF_train_data, train_label)
        test_results = RF.predict(test_data)
        print("RF classifier accuracy:", metrics.accuracy_score(test_label, test_results))
        np.save('RF_classifier_acc' + datetime.datetime.now().strftime("%m_%d_%y_%H%M"), np.array(metrics.accuracy_score(test_label, test_results)))
    
    # Plot an ROC if desired
    if args.plot_ROC:
        legend = []
        print('Plotting an ROC curve with threshold increments of 0.05')
        count = 0
        if args.check_RF:
            RF_train_data = []
            train_label = []
            for episode, episode_label in zip(pick_data['train_state'], pick_data['train_label']):
                RF_train_data.append(episode[-1])
                train_label.append(episode_label[-1])
            test_data = []
            test_label = []
            for episode, episode_label in zip(pick_data['test_state'], pick_data['test_label']):
                test_data.append(episode[-1])
                test_label.append(episode_label[-1])
            RF = RandomForestClassifier(n_estimators=100, random_state=42)
            RF.fit(RF_train_data, train_label)
            test_results = RF.predict(test_data)
            print("RF classifier accuracy:", metrics.accuracy_score(test_label, test_results))
            svc_disp = metrics.RocCurveDisplay.from_estimator(RF, test_data, test_label)
            legend.append('RF classifier')
        else:
            ROC_plot = plt.figure(figure_count)
        for policy in loaded_classifiers:
            accs = []
            TPs = [1]
            FPs = [1]
            flag=False
            for i in range(21):
                try:
                    acc, TP, FP = policy.evaluate(threshold=i * 0.05, current=flag)
                except RuntimeError:
                    flag = True
                    acc, TP, FP = policy.evaluate(threshold=i * 0.05, current=True)
                print(f'threshold {i} has tp-fp {TP}-{FP} and acc {acc}')
                TPs.append(TP)
                FPs.append(FP)
                accs.append(acc)
            print(f'policy {count} finished')
            plt.plot(FPs, TPs)
            legend.append(policy.identifier)
            count += 1
            print('best accuracy is ', np.max(accs))
        baseline = [0, 1]
        legend.append('Random Baseline')
        plt.plot(baseline, baseline, linestyle='--')
        plt.legend(legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC')

        figure_count += 1

    # Find validation ROC if desired
    if args.validate:
        legend = []
        print('Plotting an ROC curve with threshold increments of 0.05')
        count = 0
        if args.check_RF:
            RF_train_data = []
            train_label = []
            for episode, episode_label in zip(pick_data['train_state'], pick_data['train_label']):
                RF_train_data.append(episode[-1])
                train_label.append(episode_label[-1])
            test_data = []
            test_label = []
            for episode, episode_label in zip(validation_dataset['test_state'], validation_dataset['test_label']):
                test_data.append(episode[-1])
                test_label.append(episode_label[-1])
            RF = RandomForestClassifier(n_estimators=100, random_state=42)
            RF.fit(RF_train_data, train_label)
            test_results = RF.predict(test_data)
            print("RF classifier accuracy:", metrics.accuracy_score(test_label, test_results))
            svc_disp = metrics.RocCurveDisplay.from_estimator(RF, test_data, test_label)
            legend.append('RF classifier')
        else:
            ROC_plot = plt.figure(figure_count)
        for policy in loaded_classifiers:
            accs = []
            TPs = [1]
            FPs = [1]
            flag = False
            policy.load_dataset(train_data, validation_data)
            for i in range(21):
                try:
                    acc, TP, FP = policy.evaluate(threshold=i * 0.05, current=flag)
                except RuntimeError:
                    flag = True
                    acc, TP, FP = policy.evaluate(threshold=i * 0.05, current=True)
                print(f'threshold {i} has tp-fp {TP}-{FP} and acc {acc}')
                TPs.append(TP)
                FPs.append(FP)
                accs.append(acc)
            print(f'policy {count} finished')
            plt.plot(FPs, TPs)
            legend.append(policy.identifier)
            count += 1
            print('best accuracy is ', np.max(accs))
        baseline = [0, 1]
        legend.append('Random Baseline')
        plt.plot(baseline, baseline, linestyle='--')
        plt.legend(legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC')

        figure_count += 1

    # Visualize all plots made
    print('Displaying all plots other than single episode')
    plt.show()

    # Plot single example if desired
    if args.plot_example:
        legend = ['Z Force', 'Correct Output']
        print('Plotting single episode')
        count = 0
        single_episode_flag = True
        while single_episode_flag:
            single_episode_plot = plt.figure(figure_count)
            plot_bounds = [loaded_classifiers[0].visualization_range[0], loaded_classifiers[0].visualization_range[1]]
            first_plot = True
            for policy in loaded_classifiers:
                x, y, outputs = policy.evaluate_episode()
                if first_plot:
                    plt.plot(range(len(x)), x,
                             linewidth=2, c='red')
                    plt.scatter(
                        range(len(y)), y,
                        c='green')
                    first_plot = False
                    legend.append(policy.identifier + ' raw output')
                    legend.append(policy.identifier + ' rounded output')
                count += 1
                rounded_outputs = []
                for i in range(len(outputs)):
                    rounded_outputs.append((outputs[i] > 0.5) + 0.05)
                plt.scatter(range(len(outputs)), outputs, marker="+")
                plt.scatter(range(len(rounded_outputs)), rounded_outputs, marker="x")
            plt.legend(legend)
            plt.show()
            figure_count += 1
            flag_choice = input('generate another plot? y/n')
            if flag_choice.lower() == 'n':
                single_episode_flag = False
            count += 1
            figure_count += 1

    # Perform ablation on feature groups if desired
    if args.ablate:
        perform_ablation(pick_data, args)
