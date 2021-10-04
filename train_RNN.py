#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:45:34 2021

@author: orochi
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from RNNs import GRUNet, LSTMNet
import pickle as pkl
import datetime


def train(train_loader, test_loader, eval_points=None, epochs=5, model_type='GRU', layers=4, hidden=100, output='grasp',
          drop_prob=0.2, input_dim=None, train_points=0):
    # Set hyperparameters
    if input_dim == None:
        input_dim = 51
    if output == 'grasp':
        output_dim = 1
        loss_fn = nn.MSELoss()
        network_type = 1
    elif output == 'slip':
        output_dim = 1
        loss_fn = nn.MSELoss()
        network_type = 1
    elif output == 'contact':
        output_dim = 6
        loss_fn = nn.MSELoss()
        network_type = 2
    elif output == 'drop':
        output_dim = 1
        loss_fn = nn.MSELoss()
        network_type = 1
    batch_size = 5000
    # Instantiate the models
    if model_type == 'GRU':
        model = GRUNet(input_dim, hidden, output_dim, layers, drop_prob)
        model_copy = copy.deepcopy(model)
        backup_acc = 0

    elif model_type == 'LSTM':
        model = LSTMNet(input_dim, hidden, output_dim, layers, drop_prob)
        model_copy = copy.deepcopy(model)
        backup_acc = 0
    # Define loss function and optimizer
    if torch.cuda.is_available():
        model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    accs = []
    TP_rate = []
    FP_rate = []
    losses = []
    steps = []
    print('starting training, finding the starting accuracy for random model of type', output)
    acc, TP, FP = evaluate(model, test_loader, eval_points, network_type, input_dim)
    print(f'starting: accuracy - {acc}, TP rate - {TP}, FP rate - {FP}')
    accs.append(acc)
    TP_rate.append(TP)
    FP_rate.append(FP)
    losses.append(0)
    steps.append(0)
    net_loss = 0
    for epoch in range(1, epochs + 1):
        hiddens = model.init_hidden(batch_size)
        net_loss = 0
        epoch_loss = 0
        step = 0
        for x, label in train_loader:
            x = torch.reshape(x, (5000, 1, input_dim))
            if model_type == "GRU":
                hiddens = hiddens.data
            else:
                hiddens = tuple([e.data for e in hiddens])
            pred, hiddens = model(x.to(device).float(), hiddens)
            for param in model.parameters():
                if torch.isnan(param).any():
                    print('shit went sideways')
            if network_type == 1:
                pred = torch.reshape(pred, (5000,))
                loss = loss_fn(pred, label.to(device).float())
            else:
                loss = loss_fn(pred.to('cpu'), label.to('cpu').float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            net_loss += float(loss)
            epoch_loss += float(loss)
            step += 1
        acc, TP, FP = evaluate(model, test_loader, eval_points, network_type, input_dim)
        print(f'epoch {epoch}: accuracy - {acc}, loss - {epoch_loss}, TP rate - {TP}, FP rate - {FP}')
        if acc > backup_acc:
            model_copy = copy.deepcopy(model)
            backup_acc = acc
        accs.append(acc)
        losses.append(net_loss)
        steps.append(epoch)
        TP_rate.append(TP)
        FP_rate.append(FP)
        net_loss = 0
    print(f'returning best recorded model with acc = {backup_acc}')
    return model_copy, accs, losses, steps, TP_rate, FP_rate


def perform_ablation(reduced_train_state, reduced_test_state, reduced_train_label, reduced_test_label):
    labels = {'IMU Accelearation': [0, 1, 2, 9, 10, 11, 18, 19, 20], 'IMU Velocity': [3, 4, 5, 12, 13, 14, 21, 22, 23],
              'Joint Pos': [6, 15, 24],
              'Joint Velocity': [7, 16, 25], 'Joint Effort': [8, 17, 26], 'Arm Joint State': [27, 28, 29, 30, 31, 32],
              'Arm Joint Velocity': [33, 34, 35, 36, 37, 38], 'Arm Joint Effort': [39, 40, 41, 42, 43, 44],
              'Wrench Force': [45, 46, 47],
              'Wrench Torque': [48, 49, 50]}
    full_list = np.array(range(51))
    missing_labels = []
    missing_names = ''
    worst_names = []
    sizes = [51]
    reduced_train_data_grasp_success = TensorDataset(torch.from_numpy(reduced_train_state),
                                                     torch.from_numpy(reduced_train_label[:, -2]))
    reduced_test_data_grasp_success = TensorDataset(torch.from_numpy(reduced_test_state),
                                                    torch.from_numpy(reduced_test_label[:, -2]))
    reduced_train_loader_grasp_success = DataLoader(reduced_train_data_grasp_success, shuffle=False,
                                                    batch_size=batch_size, drop_last=True)
    reduced_test_loader_grasp_success = DataLoader(reduced_test_data_grasp_success, shuffle=False,
                                                   batch_size=batch_size, drop_last=True)
    performance = []
    for i in range(3):
        trained_grasp_lstm, best_accuracies, losses, steps, TP_rate, FP_rate = train(reduced_train_loader_grasp_success,
                                                                                     reduced_test_loader_grasp_success,
                                                                                     len(reduced_test_state), epochs=60,
                                                                                     model_type='LSTM', output='grasp',
                                                                                     train_points=len(
                                                                                         reduced_train_state))
        performance = np.max(best_accuracies)
        print('model finished, saving now')
        torch.save(trained_grasp_lstm, 'grasp_lstm_all' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pt')
        grasp_lstm_dict = {'acc': best_accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
        file = open('grasp_lstm_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
        pkl.dump(grasp_lstm_dict, file)
    file.close()
    best_accuracies = [np.average(performance)]
    best_acc_std_dev = np.std(best_accuracies)
    print(best_accuracies)
    for phase in range(9):
        big_accuracies = []
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
            reduced_train_data_grasp_success = TensorDataset(torch.from_numpy(reduced_train_state[:, used_labels]),
                                                             torch.from_numpy(reduced_train_label[:, -2]))
            reduced_test_data_grasp_success = TensorDataset(torch.from_numpy(reduced_test_state[:, used_labels]),
                                                            torch.from_numpy(reduced_test_label[:, -2]))
            reduced_train_loader_grasp_success = DataLoader(reduced_train_data_grasp_success, shuffle=False,
                                                            batch_size=batch_size, drop_last=True)
            reduced_test_loader_grasp_success = DataLoader(reduced_test_data_grasp_success, shuffle=False,
                                                           batch_size=batch_size, drop_last=True)
            performance = []
            for i in range(3):
                trained_grasp_lstm, accuracies, losses, steps, TP_rate, FP_rate = train(
                    reduced_train_loader_grasp_success, reduced_test_loader_grasp_success, len(reduced_test_state),
                    epochs=60, model_type='LSTM', output='grasp', input_dim=len(reduced_train_state[0, used_labels]))
                performance.append(np.max(accuracies))
                print('model finished, saving now')
                torch.save(trained_grasp_lstm, 'grasp_lstm_' + missing_names + name + datetime.datetime.now().strftime(
                    "%m_%d_%y_%H%M") + '.pt')
                grasp_lstm_dict = {'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
                file = open('grasp_lstm_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
                pkl.dump(grasp_lstm_dict, file)
                file.close()
            big_accuracies.append(np.average(performance))
            acc_std_dev.append(np.std(performance))
            names.append(name)
            indexes.append(missing_label)
        best_one = np.argmax(big_accuracies)
        print('best thing', best_one, np.shape(big_accuracies), np.shape(names))
        missing_names = missing_names + names[best_one]
        missing_labels.extend(indexes[best_one])
        best_accuracies.append(big_accuracies[best_one])
        best_acc_std_dev.append(acc_std_dev[best_one])
        worst_names.append(names[best_one])
        sizes.append(sizes[phase] - len(labels[names[best_one]]))
        labels.pop(names[best_one])
    print('ablation finished. best accuracies throughout were', best_accuracies)
    print('names removed in this order', worst_names)

    grasp_ablation_dict = {'num inputs': sizes, 'best accuracy': best_accuracies, 'std dev': best_acc_std_dev,
                           'names': worst_names}
    file = open('grasp_ablation_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
    pkl.dump(grasp_ablation_dict, file)
    file.close()

