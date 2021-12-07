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
from torch.utils.data import TensorDataset, DataLoader
from utils import RNNDataset
from itertools import islice


def make_data_name(model_name):
    ind = model_name.find('model')
    return model_name[0:ind] + 'data' + model_name[ind + 5:]


def build_dataset(database, args):
    #TODO add in functionality to change the batch size
    train_dataset = RNNDataset(database['train_state'], database['train_label'], 1)
    test_dataset = RNNDataset(database['test_state'], database['test_label'], 1)
    # loader = DataLoader(iterable_dataset, batch_size=None)

    # train_trim_inds = database['train_reduce_inds']
    # test_trim_inds = database['test_reduce_inds']
    # if args.reduced:
    #     train_state = database['train_state'][train_trim_inds == 1, :]
    #     test_state = database['test_state'][test_trim_inds == 1, :]
    #     train_label = database['train_label'][train_trim_inds == 1, :]
    #     test_label = database['test_label'][test_trim_inds == 1, :]
    # else:
    #     train_state = database['train_state']
    #     test_state = database['test_state']
    #     train_label = database['train_label']
    #     test_label = database['test_label']
    #
    # if args.goal.lower() == 'grasp':
    #     train_label = train_label[:, -2]
    #     test_label = test_label[:, -2]
    # elif args.goal.lower() == 'contact':
    #     train_label = train_label[:, 0:6]
    #     test_label = test_label[:, 0:6]
    # elif args.goal.lower() == 'slip':
    #     train_label = train_label[:, -3]
    #     test_label = test_label[:, -3]
    # elif args.goal.lower() == 'drop':
    #     train_label = train_label[:, -1]
    #     test_label = test_label[:, -1]
    # print(np.count_nonzero(test_label))
    # train_dataset = TensorDataset(torch.from_numpy(train_state), torch.from_numpy(train_label))
    # test_dataset = TensorDataset(torch.from_numpy(test_state), torch.from_numpy(test_label))
    return train_dataset, test_dataset


def setup_args(args=None):
    """ Set important variables based on command line arguments OR passed on argument values
    returns: Full set of arguments to be parsed"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="LSTM", type=str)  # RNN used
    parser.add_argument("--epochs", default=25, type=int)  # num epochs trained for
    parser.add_argument("--layers", default=4, type=int)  # num layers in RNN
    parser.add_argument("--hiddens", default=100, type=int)  # num hidden nodes per layer
    parser.add_argument("--drop_prob", default=0.2, type=float)  # drop probability
    parser.add_argument("--reduced", default=True, type=bool)  # flag indicating to reduce data to only grasp part
    parser.add_argument("--data_path",
                        default="/media/avl/StudyData/Apple Pick Data/Apple Proxy Picks/2 - Apple Proxy with spherical approach/",
                        type=str)  # path to unprocessed data
    parser.add_argument("--policy", default=None, type=str)  # filepath to trained policy
    parser.add_argument("--ablate", default=False, type=bool)  # flag to determine if ablation should be run
    parser.add_argument("--plot_acc", default=False,
                        type=bool)  # flag to determine if we want to plot the eval acc over epochs
    parser.add_argument("--plot_loss", default=False,
                        type=bool)  # flag to determine if we want to plot the loss over epochs
    parser.add_argument("--plot_ROC", default=False, type=bool)  # flag to determine if we want to plot an ROC
    parser.add_argument("--plot_TP_FP", default=False,
                        type=bool)  # flag to determine if we want to plot the TP and FP over epochs
    parser.add_argument("--plot_example", default=False,
                        type=bool)  # flag to determine if we want to plot an episode and the networks perfromance
    parser.add_argument("--goal", default="grasp", type=str)  # desired output of the lstm
    parser.add_argument("--reprocess", default=False, type=bool)  # bool to manually reprocess data if changed
    parser.add_argument("--compare_policy", default=False,
                        type=bool)  # bool to train new policy and compare to old policy in all plots
    parser.add_argument("--batch_size", default=5000, type=int)  # number of points in a batch during training

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Read in arguments from command line
    args = setup_args()

    # Load processed data if it exists and process csvs if it doesn't
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
    # Load processed data into dataset as required by goal argument
    train_data, test_data = build_dataset(pick_data, args)

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
        classifier.load_model_data(make_data_name(args.policy))
    loaded_classifiers.append(classifier)
    loaded_dicts.append(classifier.get_data_dict())
    if args.compare_policy:
        old_classifier = AppleClassifier(train_data, test_data, vars(args))
        old_classifier.load_model(args.policy)
        old_classifier.load_model_data(make_data_name(args.policy))
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
            legend.append(data['ID'])
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

    # Plot an ROC if desired
    if args.plot_ROC:
        legend = []
        print('Plotting an ROC curve with threshold increments of 0.05')
        count = 0
        ROC_plot = plt.figure(figure_count)
        for policy in loaded_classifiers:
            accs = []
            TPs = [1]
            FPs = [1]
            for i in range(21):
                _, TP, FP = policy.evaluate_with_delay(threshold=i * 0.05)
                TPs.append(TP)
                FPs.append(FP)
            print(f'policy {count} finished')
            plt.plot(FPs, TPs)
            legend.append(policy.identifier)
            count += 1
        baseline = [0, 1]
        legend.append('Random Baseline')
        plt.plot(baseline, baseline, linestyle='--')
        plt.legend(legend)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
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
            plt.plot(range(plot_bounds[0], plot_bounds[1]),
                     test_data.tensors[0][
                     plot_bounds[0]:plot_bounds[1],
                     -4] / 20,
                     linewidth=2, c='red')
            plt.scatter(
                range(plot_bounds[0], plot_bounds[1]),
                test_data.tensors[1][
                plot_bounds[0]:plot_bounds[1]],
                c='green')
            for policy in loaded_classifiers:
                outputs = policy.evaluate_secondary()
                legend.append(policy.identifier + ' raw output')
                legend.append(policy.identifier + ' rounded output')
                count += 1
                rounded_outputs = []
                for i in range(len(outputs)):
                    rounded_outputs.append((outputs[i] > 0.5) + 0.05)
                plt.scatter(range(plot_bounds[0], plot_bounds[1]), outputs, marker="+")
                plt.scatter(range(plot_bounds[0], plot_bounds[1]), rounded_outputs, marker="x")
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
        perform_ablation(train_data, test_data, args)
