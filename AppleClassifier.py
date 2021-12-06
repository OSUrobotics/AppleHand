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
import time


class AppleClassifier():
    def __init__(self, train_dataset, test_dataset, param_dict, model=None):
        """
        A class for training, evaluating and generating plots of a classifier
        for apple pick data
        @param train_loader - DataLoader with training data loaded
        @param test_loader - DataLoader with testing data loaded
        @param param_dict - Dictionary with parameters for model, training, etc
        for detail on acceptable parameters, look at valid_parameters.txt
        """
        try:
            self.epochs = param_dict['epochs']
        except KeyError:
            self.epochs = 25
        try:
            self.model_type = param_dict['model_type']
        except KeyError:
            self.model_type = "LSTM"
        try:
            self.layers = param_dict['layers']
        except KeyError:
            self.layers = 4
        try:
            self.hidden = param_dict['hidden']
        except KeyError:
            self.hidden = 100
        try:
            self.drop_prob = param_dict['drop_prob']
        except KeyError:
            self.drop_prob = 0.2
        try:
            self.batch_size = min(param_dict['batch_size'], len(test_dataset))
        except KeyError:
            self.batch_size = min(5000, len(test_dataset))
        try:
            self.input_dim = param_dict['input_dim']
        except KeyError:
            self.input_dim = len(train_dataset[0][0])
        try:
            self.outputs = param_dict['outputs']
        except KeyError:
            self.outputs = "grasp"

        if self.outputs == 'grasp':
            self.output_dim = 1
            self.loss_fn = nn.MSELoss()
            self.network_type = 1
        elif self.outputs == 'slip':
            self.output_dim = 1
            self.loss_fn = nn.MSELoss()
            self.network_type = 1
        elif self.outputs == 'contact':
            self.output_dim = 6
            self.loss_fn = nn.MSELoss()
            self.network_type = 2
        elif self.outputs == 'drop':
            self.output_dim = 1
            self.loss_fn = nn.MSELoss()
            self.network_type = 1

        self.train_data = DataLoader(train_dataset, shuffle=False,
                                     batch_size=self.batch_size,
                                     drop_last=True)
        self.test_data = DataLoader(test_dataset, shuffle=False,
                                    batch_size=self.batch_size,
                                    drop_last=True)

        self.accuracies = []
        self.TP_rate = []
        self.FP_rate = []
        self.losses = []
        self.steps = []
        self.visualization_range = [0, 1000]
        # Instantiate the model
        if model is not None:
            self.model = []
            self.load_model(model)
        else:
            if self.model_type == 'GRU':
                self.model = GRUNet(self.input_dim, self.hidden, self.output_dim,
                                    self.layers, self.drop_prob)
            elif self.model_type == 'LSTM':
                self.model = LSTMNet(self.input_dim, self.hidden, self.output_dim,
                                     self.layers, self.drop_prob)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.best_model = copy.deepcopy(self.model)

        self.test_size = len(test_dataset)
        self.train_size = len(train_dataset)
        self.identifier = self.outputs
        self.generate_ID()

    def load_model(self, filepath):
        self.model = torch.load('./models/'+filepath + '.pt')
        self.model.eval()

    def load_model_data(self, filepath):
        file = open('./data/'+filepath + '.pkl', 'rb')
        temp_dict = pkl.load(file)
        file.close()
        self.accuracies = temp_dict['acc']
        self.TP_rate = temp_dict['TP']
        self.FP_rate = temp_dict['FP']
        self.losses = temp_dict['loss']
        self.steps = temp_dict['steps']

    def get_data_dict(self):
        classifier_dict = {'acc': self.accuracies, 'loss': self.losses,
                           'steps': self.steps, 'TP': self.TP_rate,
                           'FP': self.FP_rate, 'ID': self.identifier}
        return classifier_dict.copy()

    def generate_ID(self):
        if self.batch_size != 5000:
            self.identifier = self.identifier + '_batch=' +str(self.batch_size)
        if self.epochs != 25:
            self.identifier = self.identifier +'_epochs='+str(self.epochs)
        if self.hidden != 100:
            self.identifier = self.identifier + '_hidden_nodes=' + str(self.hidden)
        if self.input_dim !=51:
            self.identifier = self.identifier + '_input_size=' + str(self.input_dim)
        if self.layers != 4:
            self.identifier = self.identifier + '_hidden_layers=' + str(self.layers)
        if self.drop_prob != 0.2:
            self.identifier = self.identifier + '_drop_probability=' + str(self.drop_prob)

    def train(self):
        backup_acc = 0
        # Define loss function and optimizer
        if torch.cuda.is_available():
            self.model.cuda()
        optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.model.train()
        print('starting training, finding the starting accuracy for random model of type', self.outputs)
        acc, TP, FP = self.evaluate(0.5)
        print(f'starting: accuracy - {acc}, TP rate - {TP}, FP rate - {FP}')
        self.accuracies.append(acc)
        self.TP_rate.append(TP)
        self.FP_rate.append(FP)
        self.losses.append(0)
        self.steps.append(0)
        net_loss = 0
        for epoch in range(1, self.epochs + 1):
            hiddens = self.model.init_hidden(self.batch_size)
            net_loss = 0
            epoch_loss = 0
            step = 0
            for x, label in self.train_data:
                x = torch.reshape(x, (self.batch_size, 1, self.input_dim))
                if self.model_type == "GRU":
                    hiddens = hiddens.data
                else:
                    hiddens = tuple([e.data for e in hiddens])
                pred, hiddens = self.model(x.to(self.device).float(), hiddens)
                for param in self.model.parameters():
                    if torch.isnan(param).any():
                        print('shit went sideways')
                if self.network_type == 1:
                    pred = torch.reshape(pred, (self.batch_size,))
                    loss = self.loss_fn(pred, label.to(self.device).float())
                else:
                    loss = self.loss_fn(pred.to('cpu'), label.to('cpu').float())
                optim.zero_grad()
                loss.backward()
                optim.step()
                net_loss += float(loss)
                epoch_loss += float(loss)
                step += 1
            acc, TP, FP = self.evaluate(0.5)
            print(f'epoch {epoch}: accuracy - {acc}, loss - {epoch_loss}, TP rate - {TP}, FP rate - {FP}')
            if acc > backup_acc:
                self.best_model = copy.deepcopy(self.model)
                backup_acc = acc
            self.accuracies.append(acc)
            self.losses.append(net_loss)
            self.steps.append(epoch)
            self.TP_rate.append(TP)
            self.FP_rate.append(FP)
            net_loss = 0
        print(f'Finished training, best recorded model had acc = {backup_acc}')

    def evaluate(self, threshold=0.5):
        #        start = time.time()
        self.model.eval()
        hidden_layer = self.model.init_hidden(self.batch_size)
        if self.network_type == 'aa':
            last_ind = 3
            outputs = np.zeros((self.test_size, 3))
            test_labels = np.zeros((self.test_size, 3))
        elif self.network_type == 1:
            last_ind = 1
            outputs = np.zeros((self.test_size, 1))
            test_labels = np.zeros((self.test_size, 1))
        elif self.network_type == 2:
            last_ind = 6
            outputs = np.zeros((self.test_size, 6))
            test_labels = np.zeros((self.test_size, 6))
        else:
            last_ind = 4
            outputs = np.zeros((self.test_size, 4))
            test_labels = np.zeros((self.test_size, 4))
        acc = 0
        count = 0
        for x, y in self.test_data:
            x = torch.reshape(x, (self.batch_size, 1, self.input_dim))
            y = torch.reshape(y, (self.batch_size, last_ind))
            hidden_layer = tuple([e.data for e in hidden_layer])
            out, hidden_layer = self.model(x.to(self.device).float(), hidden_layer)
            mod = len(out)
            count += mod
            outputs[count - mod:count] = out.to('cpu').detach().numpy()
            test_labels[count - mod:count] = y.to('cpu').detach().numpy()
        if self.network_type == 'g':
            for i in range(len(outputs)):
                temp = abs(np.argmax(outputs[i].to('cpu').detach().numpy()) - test_labels[i])
                if temp >= 1:
                    acc += 0
                else:
                    acc += 1
        elif self.network_type == 1:
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
        elif self.network_type == 2:
            output_data = outputs.to('cpu').detach().numpy()
            test_labels = test_labels.to('cpu').detach().numpy()
            output_data = output_data.astype(int)
            output_data = output_data > threshold
            temp = output_data - test_labels
            FP_diffs = np.count_nonzero(temp > 0)
            FN_diffs = np.count_nonzero(temp < 0)
            num_pos = np.count_nonzero(test_labels)
            num_neg = self.test_size * 6 - num_pos
            FP = FP_diffs / num_neg
            FN = FN_diffs / num_pos
            TP = 1 - FN
            diffs = np.count_nonzero(temp)
            acc = 1 - diffs / 6 / len(output_data)
        else:
            for i in range(len(outputs)):
                acc += sum(abs(outputs[i].to('cpu').detach().numpy()[0] - test_labels[i]))
                acc += min(temp, abs(temp - 2 * 3.14159))
        self.model.train()
        return acc, TP, FP

    def evaluate_with_delay(self, threshold=0.5):
        #        start = time.time()
        self.model.eval()
        hidden_layer = self.model.init_hidden(self.batch_size)
        if self.network_type == 'aa':
            last_ind = 3
            outputs = np.zeros((self.test_size, 3))
            test_labels = np.zeros((self.test_size, 3))
        elif self.network_type == 1:
            last_ind = 1
            outputs = np.zeros((self.test_size, 1))
            test_labels = np.zeros((self.test_size, 1))
        elif self.network_type == 2:
            last_ind = 6
            outputs = np.zeros((self.test_size, 6))
            test_labels = np.zeros((self.test_size, 6))
        else:
            last_ind = 4
            outputs = np.zeros((self.test_size, 4))
            test_labels = np.zeros((self.test_size, 4))
        acc = 0
        count = 0
        for x, y in self.test_data:
            x = torch.reshape(x, (self.batch_size, 1, self.input_dim))
            y = torch.reshape(y, (self.batch_size, last_ind))
            hidden_layer = tuple([e.data for e in hidden_layer])
            out, hidden_layer = self.model(x.to(self.device).float(), hidden_layer)
            mod = len(out)
            count += mod
            outputs[count - mod:count] = out.to('cpu').detach().numpy()
            test_labels[count - mod:count] = y.to('cpu').detach().numpy()
        if self.network_type == 'g':
            for i in range(len(outputs)):
                temp = abs(np.argmax(outputs[i].to('cpu').detach().numpy()) - test_labels[i])
                if temp >= 1:
                    acc += 0
                else:
                    acc += 1
        elif self.network_type == 1:
            output_data = outputs[:, 0]
            test_labels = test_labels[:, 0]
            output_data[124:] = AppleClassifier.moving_average(output_data, 125)
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
        elif self.network_type == 2:
            output_data = outputs.to('cpu').detach().numpy()
            test_labels = test_labels.to('cpu').detach().numpy()
            output_data = output_data.astype(int)
            output_data = output_data > threshold
            temp = output_data - test_labels
            FP_diffs = np.count_nonzero(temp > 0)
            FN_diffs = np.count_nonzero(temp < 0)
            num_pos = np.count_nonzero(test_labels)
            num_neg = self.test_size * 6 - num_pos
            FP = FP_diffs / num_neg
            FN = FN_diffs / num_pos
            TP = 1 - FN
            diffs = np.count_nonzero(temp)
            acc = 1 - diffs / 6 / len(output_data)
        else:
            for i in range(len(outputs)):
                acc += sum(abs(outputs[i].to('cpu').detach().numpy()[0] - test_labels[i]))
                acc += min(temp, abs(temp - 2 * 3.14159))
        self.model.train()
        return acc, TP, FP

    def evaluate_secondary(self):
        self.model.eval()
        hidden_layer = self.model.init_hidden(self.batch_size)
        outputs = []
        last_ind = 1
        for x, y in self.test_data:
            x = torch.reshape(x, (self.batch_size, 1, self.input_dim))
            y = torch.reshape(y, (self.batch_size, last_ind))
            hidden_layer = tuple([e.data for e in hidden_layer])
            out, hidden_layer = self.model(x.to(self.device).float(), hidden_layer)
            outputs.append(out.to('cpu').detach().numpy())
        outputs = [item for sublist in outputs for item in sublist]
        outputs = np.reshape(outputs, len(outputs))
        outputs[124:] = AppleClassifier.moving_average(outputs, 125)
        outputs = outputs[self.visualization_range[0]:self.visualization_range[1]]
        self.visualization_range[0] += 1000
        self.visualization_range[1] += 1000
        self.model.train()
        return outputs

    @staticmethod
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    def save_model(self, filename=None):
        if filename is None:
            filename = './models/' + self.identifier + \
                       datetime.datetime.now().strftime("%m_%d_%y_%H%M")
        torch.save(self.best_model, filename + '.pt')

    def save_data(self, filename=None):
        if filename is None:
            filename = './data/' + self.identifier + \
                       datetime.datetime.now().strftime("%m_%d_%y_%H%M")
        classifier_dict = {'acc': self.accuracies, 'loss': self.losses,
                           'steps': self.steps, 'TP': self.TP_rate,
                           'FP': self.FP_rate, 'ID': self.identifier}
        file = open(filename + '.pkl', 'wb')
        pkl.dump(classifier_dict, file)
