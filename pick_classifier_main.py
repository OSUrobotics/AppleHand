#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:54:09 2021

@author: orochi
"""

import copy
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from RNNs import GRUNet, LSTMNet
import pickle as pkl
import datetime
from train_RNN import train, perform_ablation
from csv_process import process_data
import argparse
from evaluate_RNN import evaluate_with_delay, evaluate_secondary


def build_dataloaders(database, args):
    train_trim_inds = database['train_reduce_inds']
    test_trim_inds = database['test_reduce_inds']
    if args.reduced:
        train_state = database['train_state'][train_trim_inds == 1, :]
        test_state = database['test_state'][test_trim_inds == 1, :]
        train_label = database['train_label'][train_trim_inds == 1, :]
        test_label = database['test_label'][test_trim_inds == 1, :]
    else:
        train_state = database['train_state']
        test_state = database['test_state']
        train_label = database['train_label']
        test_label = database['test_label']

    if goal.lower() == 'grasp':
        train_label = train_label[:, -2]
        test_label = test_label[:, -2]
    elif goal.lower() == 'contact':
        train_label = train_label[:, 0:6]
        test_label = test_label[:, 0:6]
    elif goal.lower() == 'slip':
        train_label = train_label[:, -3]
        test_label = test_label[:, -3]
    elif goal.lower() == 'drop':
        train_label = train_label[:, -1]
        test_label = test_label[:, -1]

    train_data = TensorDataset(torch.from_numpy(train_state), torch.from_numpy(train_label))
    test_data = TensorDataset(torch.from_numpy(test_state), torch.from_numpy(test_label))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, drop_last=True)

    return train_loader, test_loader


def setup_args(args=None):
    """ Set important variables based on command line arguments OR passed on argument values
    returns: Full set of arguments to be parsed"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="LSTM", type=str) # RNN used
    parser.add_argument("--epochs", default=5) # num epochs trained for
    parser.add_argument("--layers", default=4, type=int) # num layers in RNN
    parser.add_argument("--hiddens", default=100, type=int) # num hidden nodes per layer
    parser.add_argument("--drop_prob", default=0.2, type=float) # drop probability
    parser.add_argument("--reduced", default=True, type=bool) # flag indicating to reduce data to only grasp part
    parser.add_argument("--data_path", default="/media/avl/StudyData/ApplePicking Data/2 - Apple Proxy with spherical approach/", type=str)          # path to unprocessed data
    parser.add_argument("--policy", default=None, type=str) # filepath to trained policy
    parser.add_argument("--ablate", default=False, type=bool) # flag to determine if ablation should be run
    parser.add_argument("--plot_acc", default=False, type=bool) # flag to determine if we want to plot the eval acc over epochs
    parser.add_argument("--plot_loss", default=False, type=bool) # flag to determine if we want to plot the loss over epochs
    parser.add_argument("--plot_ROC", default=False, type=bool) # flag to determine if we want to plot an ROC
    parser.add_argument("--plot_TP_FP", default=False, type=bool) # flag to determine if we want to plot the TP and FP over epochs
    parser.add_argument("--plot_example", default=False, type=bool) # flag to determine if we want to plot an episode and the networks perfromance
    parser.add_argument("--goal", default="grasp", type=str) # desired output of the lstm
    parser.add_argument("--reprocess", default=False, type=bool) # bool to manually reprocess data if changed
    parser.add_argument("--compare_policy", default=False, type=bool) # bool to train new policy and compare to old policy in all plots
    parser.add_argument("--batch_size", default=5000, type=bool) # number of points in a batch during training

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Read in arguments from command line
    args = setup_args()

    # Load processed data if it exists and process csvs if it doesn't
    if args.reprocess:
        process_data(args.data_path)
        file = open('apple_dataset.pkl', 'rb')
        pick_data = pkl.load(data_file, file)
        file.close()
    else:
        try:
            file = open('apple_dataset.pkl', 'rb')
            pick_data = pkl.load(data_file, file)
            file.close()
        except FileNotFoundError:
            process_data(args.data_path)
            file = open('apple_dataset.pkl', 'rb')
            pick_data = pkl.load(data_file, file)
            file.close()

    # Load processed data into dataloader as required by goal argument
    train_loader, test_loader = build_dataloaders(pick_data, args)

    # Load policy if it exists, if not train a new one
    loaded_policies = []
    loaded_dicts = []
    if args.policy is None or args.compare_policy:
        trained_RNN, accuracies, losses, steps, TP_rate, FP_rate = train(train_loader, test_loader, eval_points = len(reduced_test_state), epochs=25,
                                                    model_type='LSTM', output='slip')
        print('model finished, saving now')
        name = args.model_type + "_" + args.goal + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M")
        torch.save(trained_RNN, name + '.pt')
        RNN_dict = {'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate, 'name': name}
        file = open(name + '.pkl', 'wb')
        pkl.dump(grasp_lstm_dict, file)
        file.close()
    else:
        trained_RNN = torch.load(args.policy + '.pt')
        trained_RNN.eval()
        file = open(name + '.pkl', 'rb')
        RNN_dict = pkl.load(file)
        file.close()
    loaded_policies.append(trained_RNN)
    loaded_dicts.append(RNN_dict)
    if args.compare_policy:
        old_trained_policy = torch.load(args.policy)
        old_trained_policy.eval()
        loaded_policies.append(old_trained_policy)
        file = open(name + '.pkl', 'rb')
        old_dict = pkl.load(file)
        file.close()
        loaded_dicts.append(old_dict)

    figure_count = 1
    # Plot accuracy over time if desired
    if args.plot_acc:
        print('Plotting classifier accuracy over time')
        legend = []
        for data in loaded_dicts:
            acc_plot = plt.figure(figure_count)
            plt.plot(data['steps'], data['acc'])
            legend.append(data['legend'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        figure_count += 1

    # Plot loss over time if desired
    if args.plot_loss:
        legend = []
        print('Plotting classifier loss over time')
        for data in loaded_dicts:
            acc_plot = plt.figure(figure_count)
            plt.plot(data['steps'], data['loss'])
            legend.append(data['legend'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        figure_count += 1

    # Plot TP and FP rate over time if desired
    if args.plot_TP_FP:
        legend = []
        print('Plotting true positive and false positive rate over time')
        for data in loaded_dicts:
            acc_plot = plt.figure(figure_count)
            plt.plot(data['steps'], data['TP'])
            plt.plot(data['steps'], data['FP'])
            legend.append('TP'+data['legend'])
            legend.append('FP'+data['legend'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        figure_count += 1
        
    # Plot an ROC if desired
    if args.plot_ROC:
        legend = []
        print('Plotting an ROC curve with threshold increments of 0.05')
        for policy in loaded_policies:
            for i in range(21):
                TODO finish writing this code and the code for single example
                acc, TP, FP = evaluate_with_delay(policy, reduced_test_loader, len(reduced_test_state), 1,
                                              threshold=i * 0.05)#, input_dim=33)
                acc1, TP1, FP1 = evaluate(trained_grasp_lstm, reduced_test_loader_grasp_success, len(reduced_test_state), 1, threshold=i * 0.05)#, input_dim=33)
                accs.append(acc)
                TPs.append(TP)
                FPs.append(FP)
                accs1.append(acc1)
                TPs1.append(TP1)
                FPs1.append(FP1)
            acc_plot = plt.figure(figure_count)
            plt.plot(data['steps'], data['TP'])
            plt.plot(data['steps'], data['FP'])
            legend.append('TP'+data['legend'])
            legend.append('FP'+data['legend'])
        plt.legend(legend)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        figure_count += 1
    
    # Visualize all plots made
    print('Displaying all plots other than single episode')
    plt.show()  
    
    # Plot single example if desired
    if args.plot_example:
        flag = True
        count = 0
        while flag:
            reduced_plot_state = reduced_test_state[count*5000:(count+1)*5000]
            reduced_plot_label = reduced_test_label[count*5000:(count+1)*5000]
            outputs = evaluate_secondary(trained_grasp_lstm, torch.from_numpy(reduced_plot_state), reduced_plot_label[:, -2])
            outputs = np.array(outputs)
            outputs = outputs[:, 0]
            outputs[124:] = moving_average(outputs, 125)
            rounded_outputs = []
            lstm_outputs = np.array(outputs)
            for i in range(len(lstm_outputs)):
                rounded_outputs.append((lstm_outputs[i] > t_hold) + 0.05)
            xaxis = np.array(range(len(reduced_plot_state)))/500
            plt.figure(figure_count)
            plt.plot(xaxis, reduced_plot_state[:, -4]/20, linewidth=2)
            plt.scatter(xaxis, rounded_outputs, marker="x", c='orange')
            plt.scatter(xaxis, lstm_outputs, marker="+", c='red')
            plt.scatter(xaxis, reduced_plot_label[:, -2], c='green')
            plt.legend(['Z Force', 'LSTM labels', 'LSTM Output', 'Correct Output'])
            plt.show()
            flag_choice = input('generate another plot? y/n')
            if flag_choice.lower() == 'n':
                flag = False
            count += 1
            figure_count += 1

    # Perform ablation on feature groups if desired
    if args.ablate:
        perform_ablation(reduced_train_state, reduced_test_state, reduced_train_label, reduced_test_label)

# perform_ablation(reduced_train_state, reduced_test_state, reduced_train_label, reduced_test_label)

# trained_slip_lstm, accuracies, losses, steps, TP_rate, FP_rate = train(reduced_train_loader_slip, reduced_test_loader_slip, eval_points = len(reduced_test_state), epochs=25,
#                                                     model_type='LSTM', output='slip')
# print('model 1 finished, saving now')
# torch.save(trained_slip_lstm, 'slip_lstm_' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pt')
# grasp_lstm_dict = {'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
# file = open('slip_lstm_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
# pkl.dump(grasp_lstm_dict, file)
# file.close()


# trained_grasp_lstm, accuracies, losses, steps, TP_rate, FP_rate = train(reduced_train_loader_grasp_success,reduced_test_loader_grasp_success,len(reduced_test_state),epochs=75,model_type='LSTM',output='grasp')#,input_dim=33)
# print('model 2 finished, saving now')
# torch.save(trained_grasp_lstm,'grasp_lstm_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
# grasp_lstm_dict = {'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
# file = open('grasp_lstm_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl', 'wb')
# pkl.dump(grasp_lstm_dict, file)
# file.close()
#
# trained_drop_lstm, accuracies, losses, steps, TP_rate, FP_rate = train(reduced_train_loader_drop,reduced_test_loader_drop,len(reduced_test_state),epochs=25,model_type='LSTM',output='drop')
# print('model 4 finished, saving now')
# torch.save(trained_drop_lstm, 'contact_lstm_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
# grasp_lstm_dict={'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
# file = open('drop_lstm_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
# pkl.dump(grasp_lstm_dict, file)
# file.close()
#
# trained_contact_lstm, accuracies, losses, steps, TP_rate, FP_rate = train(reduced_train_loader_contact,reduced_test_loader_contact,len(reduced_test_state),epochs=25,model_type='LSTM',output='contact')
# print('model 3 finished, saving now')
# torch.save(trained_contact_lstm, 'contact_lstm_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
# grasp_lstm_dict={'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
# file = open('contact_lstm_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl', 'wb')
# pkl.dump(grasp_lstm_dict, file)
# file.close()


# plt.plot(np.array(steps),accuracies)
# plt.plot(np.array(grasp_steps),grasp_accuracies)
# plt.legend(['LSTM','GRU'])
# plt.xlabel('Steps')
# plt.ylabel('Accuracy')
# plt.show()
#
# plt.plot(np.array(gru_steps),gru_accuracies)
# plt.plot(np.array(stem_steps),stem_accuracies)
# plt.legend(['Quaternion Loss','Axis Angle Loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Average Rotation Angle Error')
# plt.show()

#
# gru_outputs=evaluate_secondary(trained_grasp_gru,test_state,test_label)
# for i in range(len(gru_outputs)):
#    gru_outputs[i] = gru_outputs[i].to('cpu').detach().numpy()[0]
#    gru_outputs[i] = np.argmax(gru_outputs[i])
# plt.scatter(range(len(test_label)),test_label)
# plt.scatter(range(len(gru_outputs)),gru_outputs)
# plt.legend(['correct labels','gru labels'])
# plt.show()
