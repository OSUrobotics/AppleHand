#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:56:01 2021

@author: orochi
"""

import time
import numpy as np
import torch


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def evaluate(model, test_loader, num_points, network_type, input_dim=51, threshold = 0.5):
    start = time.time()
    model.eval()
    hidden_layer = model.init_hidden(5000)
    if network_type == 'aa':
        last_ind = 3
        outputs = np.zeros((num_points, 3))
        test_labels = np.zeros((num_points, 3))
    elif network_type == 1:
        last_ind = 1
        outputs = np.zeros((num_points, 1))
        test_labels = np.zeros((num_points, 1))
    elif network_type == 2:
        last_ind = 6
        outputs = np.zeros((num_points, 6))
        test_labels = np.zeros((num_points, 6))
    else:
        last_ind = 4
        outputs = np.zeros((num_points, 4))
        test_labels = np.zeros((num_points, 4))
    acc = 0
    count = 0
    for x, y in test_loader:
        x = torch.reshape(x, (5000, 1, input_dim))
        y = torch.reshape(y, (5000, last_ind))
        hidden_layer = tuple([e.data for e in hidden_layer])
        out, hidden_layer = model(x.to(device).float(), hidden_layer)
        mod = len(out)
        count += mod
        outputs[count - mod:count] = out.to('cpu').detach().numpy()
        test_labels[count - mod:count] = y.to('cpu').detach().numpy()
    if network_type == 'g':
        for i in range(len(outputs)):
            temp = abs(np.argmax(outputs[i].to('cpu').detach().numpy()) - test_labels[i])
            if temp >= 1:
                acc += 0
            else:
                acc += 1
    elif network_type == 1:
        output_data = outputs
        output_data = output_data > threshold
        output_data = output_data.astype(int)
        temp = output_data - test_labels
        FP_diffs = np.count_nonzero(temp > 0)
        FN_diffs = np.count_nonzero(temp < 0)
        num_pos = np.count_nonzero(test_labels)
        num_neg = np.count_nonzero(test_labels == 0)
        FP = FP_diffs / num_neg
        FN = FN_diffs / num_pos
        TP = 1 - FN
        diffs = np.count_nonzero(temp)
        acc = 1 - diffs / len(output_data)
    elif network_type == 2:
        output_data = outputs.to('cpu').detach().numpy()
        test_labels = test_labels.to('cpu').detach().numpy()
        output_data = output_data.astype(int)
        output_data = output_data > threshold
        temp = output_data - test_labels
        FP_diffs = np.count_nonzero(temp > 0)
        FN_diffs = np.count_nonzero(temp < 0)
        num_pos = np.count_nonzero(test_labels)
        num_neg = num_points * 6 - num_pos  # np.count_nonzero(test_labels == 0
        FP = FP_diffs / num_neg
        FN = FN_diffs / num_pos
        TP = 1 - FN
        diffs = np.count_nonzero(temp)
        acc = 1 - diffs / 6 / len(output_data)
    else:  #
        for i in range(len(outputs)):
            acc += sum(abs(outputs[i].to('cpu').detach().numpy()[0] - test_labels[i]))
            acc += min(temp, abs(temp - 2 * 3.14159))
    model.train()
    return acc, TP, FP

def evaluate_with_delay(model, test_loader, num_points, network_type, input_dim=51, threshold = 0.5):
    start = time.time()
    model.eval()
    hidden_layer = model.init_hidden(5000)
    if network_type == 'aa':
        last_ind = 3
        outputs = np.zeros((num_points, 3))
        test_labels = np.zeros((num_points, 3))
    elif network_type == 1:
        last_ind = 1
        outputs = np.zeros((num_points, 1))
        test_labels = np.zeros((num_points, 1))
    elif network_type == 2:
        last_ind = 6
        outputs = np.zeros((num_points, 6))
        test_labels = np.zeros((num_points, 6))
    else:
        last_ind = 4
        outputs = np.zeros((num_points, 4))
        test_labels = np.zeros((num_points, 4))
    acc = 0
    count = 0
    for x, y in test_loader:
        x = torch.reshape(x, (5000, 1, input_dim))
        y = torch.reshape(y, (5000, last_ind))
        hidden_layer = tuple([e.data for e in hidden_layer])
        out, hidden_layer = model(x.to(device).float(), hidden_layer)
        mod = len(out)
        count += mod
        outputs[count - mod:count] = out.to('cpu').detach().numpy()
        test_labels[count - mod:count] = y.to('cpu').detach().numpy()
    if network_type == 'g':
        for i in range(len(outputs)):
            temp = abs(np.argmax(outputs[i].to('cpu').detach().numpy()) - test_labels[i])
            if temp >= 1:
                acc += 0
            else:
                acc += 1
    elif network_type == 1:
        output_data = outputs[:,0]
        test_labels = test_labels[:,0]
        output_data[124:] = moving_average(output_data,125)
        output_data = output_data > threshold
        output_data = output_data.astype(int)
        temp = output_data - test_labels
        FP_diffs = np.count_nonzero(temp > 0)
        FN_diffs = np.count_nonzero(temp < 0)
        num_pos = np.count_nonzero(test_labels)
        num_neg = np.count_nonzero(test_labels == 0)
        FP = FP_diffs / num_neg
        FN = FN_diffs / num_pos
        TP = 1 - FN
        diffs = np.count_nonzero(temp)
        acc = 1 - diffs / len(output_data)
    elif network_type == 2:
        output_data = outputs.to('cpu').detach().numpy()
        test_labels = test_labels.to('cpu').detach().numpy()
        output_data = output_data.astype(int)
        output_data = output_data > threshold
        temp = output_data - test_labels
        FP_diffs = np.count_nonzero(temp > 0)
        FN_diffs = np.count_nonzero(temp < 0)
        num_pos = np.count_nonzero(test_labels)
        num_neg = num_points * 6 - num_pos  # np.count_nonzero(test_labels == 0
        FP = FP_diffs / num_neg
        FN = FN_diffs / num_pos
        TP = 1 - FN
        diffs = np.count_nonzero(temp)
        acc = 1 - diffs / 6 / len(output_data)
    else:  #
        for i in range(len(outputs)):
            acc += sum(abs(outputs[i].to('cpu').detach().numpy()[0] - test_labels[i]))
            acc += min(temp, abs(temp - 2 * 3.14159))
    model.train()
    return acc, TP, FP

def evaluate_secondary(model, test_data, test_labels):
    model.eval()
    hidden_layer = model.init_hidden(1)
    outputs = []
    for x, y in zip(test_data, test_labels):
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        out, hidden_layer = model(x.to(device).float(), hidden_layer)
        outputs.append(out.to('cpu').detach().numpy()[0])
    model.train()
    return outputs


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
