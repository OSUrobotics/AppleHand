#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:45:34 2021

@author: Nigel Swenson
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, SequentialSampler
from RNNs import GRUNet, LSTMNet
import pickle as pkl
import datetime
import time
from utils import unpack_arr
import sklearn.metrics as metrics


class AppleClassifier:
    def __init__(self, train_dataset, test_dataset, param_dict, model=None, validation_dataset=None):
        """
        A class for training, evaluating and generating plots of a classifier
        for apple pick data
        @param train_loader - RNNDataset with training data loaded
        @param test_loader - RNNDataset with testing data loaded
        @param param_dict - Dictionary with parameters for model, training, etc
        for detail on acceptable parameters, look at valid_parameters.txt
        """
        self.eval_type = 'last'
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
            self.hidden = param_dict['hiddens']
        except KeyError:
            self.hidden = 100
        try:
            self.drop_prob = param_dict['drop_prob']
        except KeyError:
            self.drop_prob = 0.2
        try:
            self.input_dim = param_dict['input_dim']
        except KeyError:
            print('we are here ',test_dataset.shape[2])
            self.input_dim = test_dataset.shape[2]
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

        self.batch_size = param_dict['batch_size']
        simple_test_sampler = SequentialSampler(test_dataset)
#        num_pos = 0
#        num_neg = 0
#        for state, label, lens, names in validation_dataset:
#            if label[-1] == 0:
#                num_neg += 1
#            elif label[-1] == 1:
#                num_pos += 1
#            else:
#                print('yo wtf?')
#        print('perecnt we should use is ', num_pos/(num_pos+num_neg))
        try:
            num_pos = 0
            num_neg = 0
            for state, label, lens, names in train_dataset:
                if label[-1] == 0:
                    num_neg += 1
                elif label[-1] == 1:
                    num_pos += 1
                else:
                    print('error! episode neither success or failure. setting weight to 0.5')
                    
            pos_ratio = 0.5
            neg_ratio = (1-param_dict['s_f_bal'])*num_pos*pos_ratio/(num_neg*param_dict['s_f_bal'])
        except TypeError:
            pos_ratio = 0.5
            neg_ratio = 0.5
            print('no desired s_f balance, sampling all episodes equally')
        s_f_bal = []
        for state, label, lens, names in train_dataset:
            if label[-1] == 0:
                s_f_bal.append(neg_ratio)
            elif label[-1] == 1:
                s_f_bal.append(pos_ratio)
            else:
                print('error! episode neither success or failure. setting weight to 0.5')
                s_f_bal.append(0.5)
        self.data_sampler = WeightedRandomSampler(s_f_bal, param_dict['batch_size'], replacement=False)
        self.train_data = DataLoader(train_dataset, shuffle=False,
                                     batch_size=param_dict['batch_size'], sampler=self.data_sampler)
        self.test_data = DataLoader(test_dataset, shuffle=False,
                                        batch_size=None,sampler=simple_test_sampler)
        if validation_dataset is not None:
            self.validation_data = DataLoader(validation_dataset, shuffle=False,
                                              batch_size=None)
            self.validation_size = validation_dataset.shape
            self.validation_accuracies = []
            print('validation size', self.validation_size)
        else:
            self.validation_data = None
            self.validation_size = None
            self.validation_accuracies = []
        self.train_accuracies = []
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
        self.val_model = copy.deepcopy(self.model)
        self.plot_ind = 0
        self.test_size = test_dataset.shape
        self.train_size = train_dataset.shape
        print('test and train size ', self.test_size, self.train_size)
        self.identifier = self.outputs
        self.generate_ID()
        self.label_times = []
        print('input size: ', self.input_dim)
        print('number of weights in model ', self.count_parameters())
        self.group_acc = []
        self.group_val_acc = []

    def count_parameters(self):
        """
        function to calculate number of trainable weights in model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_dataset(self, train_data, test_data):
        """
        function to load train and test data
        @params - must be RNNDatasets
        """
        self.train_data = DataLoader(train_data, shuffle=False,
                                     batch_size=None)
        self.test_data = DataLoader(test_data, shuffle=False,
                                    batch_size=None)
        self.test_size = test_data.shape
        self.train_size = train_data.shape

    def load_model(self, filepath):
        """
        function to load model
        """
        self.model = torch.load('./models/' + filepath + '.pt')
        self.model.eval()

    def load_model_data(self, filepath):
        """
        function to load training performance, useful for continuing training with a particular model and having all data saved in same location
        """
        file = open('./generated_data/' + filepath + '.pkl', 'rb')
        temp_dict = pkl.load(file)
        file.close()
        self.accuracies = temp_dict['acc']
        self.TP_rate = temp_dict['TP']
        self.FP_rate = temp_dict['FP']
        self.losses = temp_dict['loss']
        self.steps = temp_dict['steps']
        self.train_accuracies = temp_dict['train_acc']
        try:
            self.validation_accuracies = temp_dict['validation_acc']
        except KeyError:
            pass

    def get_data_dict(self):
        """
        function to generate dict of performance during training for saving
        """
        classifier_dict = {'acc': self.accuracies, 'loss': self.losses,
                           'steps': self.steps, 'TP': self.TP_rate,
                           'FP': self.FP_rate, 'ID': self.identifier,
                           'train_acc': self.train_accuracies, 'validation_acc': self.validation_accuracies}
        return classifier_dict.copy()

    def get_best_performance(self, ind_range):
        """
        function to parse ROC response from scikit.metrics into max acc and best TP/FP over a window of epochs
        @param ind_range - list containing start and end epoch to consider
        """
        if (ind_range[0] >= 0) and (ind_range[1] < 0):
            ind_range[1] = len(self.accuracies) + ind_range[1]
        max_acc_ind = [np.argmax(a) for a in self.accuracies[ind_range[0]:ind_range[1]]]
        max_acc_train = [max(a) for a in self.train_accuracies[ind_range[0]:ind_range[1]]]
        max_acc = []
        best_FP = []
        best_TP = []
        for i in range(ind_range[1] - ind_range[0]):
            max_acc.append(self.accuracies[ind_range[0] + i][max_acc_ind[i]])
            best_FP.append(self.FP_rate[ind_range[0] + i][max_acc_ind[i]])
            best_TP.append(self.TP_rate[ind_range[0] + i][max_acc_ind[i]])
        return max_acc, best_TP, best_FP, max_acc_train

    def generate_ID(self):
        """
        function to create unique identifier based on frequently changed metrics and save to self.identifier for saving data and models
        """
        if self.epochs != 25:
            self.identifier = self.identifier + '_epochs=' + str(self.epochs)
        if self.hidden != 100:
            self.identifier = self.identifier + '_hidden_nodes=' + str(self.hidden)
        if self.input_dim != 43:
            self.identifier = self.identifier + '_input_size=' + str(self.input_dim)
        if self.layers != 4:
            self.identifier = self.identifier + '_hidden_layers=' + str(self.layers)
        if self.drop_prob != 0.2:
            self.identifier = self.identifier + '_drop_probability=' + str(self.drop_prob)

    def train(self):
        """
        function to train model and save performance every epoch
        """
        # eval_period controls how frequently results are printed, NOT how frequently they get recorded
        eval_period = 10
        if torch.cuda.is_available():
            self.model.cuda()
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        print('starting training, finding the starting accuracy for random model of type', self.outputs)
        acc, TP, FP, AUC = self.evaluate(0.5)
#        train_acc, _, _, _ = self.evaluate(0.5, 'train')
        best_ind = np.argmax(acc)
        print(f'starting: accuracy - {acc[best_ind]}, TP rate - {TP[best_ind]}, FP rate - {FP[best_ind]}')
        self.accuracies.append(acc)
#        self.train_accuracies.append(train_acc)
        self.TP_rate.append(TP)
        self.FP_rate.append(FP)
        self.losses.append(0)
        self.steps.append(0)
        backup_AUC = AUC
        net_loss = 0
        val_AUC = 0
        for epoch in range(1, self.epochs + 1):
            net_loss = 0
            epoch_loss = 0
            step = 0
            t0 = time.time()
            for _ in range(int(self.train_size[0]/self.batch_size)):
                for x, label, lens, names in self.train_data:
                    temp = np.shape(x)
                    if len(temp) == 2:
                        reshape_param = temp[0]
                    elif len(temp) == 3:
                        reshape_param = temp[0]*temp[1]
                    hiddens = self.model.init_hidden(reshape_param)
                    x = torch.reshape(x, (reshape_param, 1, self.input_dim))
                    label = torch.reshape(label, (reshape_param,1))
                    
                    if self.model_type == "GRU":
                        hiddens = hiddens.data
                    else:
                        hiddens = tuple([e.data for e in hiddens])
                    pred, hiddens = self.model(x.to(self.device).float(), hiddens)
                    for param in self.model.parameters():
                        if torch.isnan(param).any():
                            print('shit went sideways')
                    if self.network_type == 1:
                        if self.eval_type == 'last':
                            loss = self.loss_fn(pred, label.to(self.device).float())
                        else:
                            loss = self.loss_fn(pred, label.to(self.device).float())
                    else:
                        loss = self.loss_fn(pred.to('cpu'), label.to('cpu').float())
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    net_loss += float(loss)
                    epoch_loss += float(loss)
                    step += 1
            t1 = time.time()
#            print('percent positive = ', counter/tot_seen)
            acc, TP, FP, AUC = self.evaluate(0.5)
#            train_acc, train_tp, train_fp, train_AUC = self.evaluate(0.5, 'train')
            if self.validation_data is not None:
                validation_acc, validation_tp, validation_fp, validation_AUC = self.evaluate(0.5, 'validation')
                self.validation_accuracies.append(validation_acc)
                if validation_AUC > val_AUC:
                    self.val_model = copy.deepcopy(self.model)
                    val_AUC = validation_AUC
            if epoch % eval_period == 0:
                # print('average time spend shuffling dataset ', self.train_data.print_times())
                max_acc, best_TP, best_FP, max_acc_train = self.get_best_performance([epoch - eval_period, epoch])
                best_epoch = np.argmax(max_acc)
                print(
                    f'epoch {epoch}: test accuracy  - {max_acc[best_epoch]}, loss - {sum(self.losses[epoch - eval_period:epoch])}, TP rate - {best_TP[best_epoch]}, FP rate - {best_FP[best_epoch]}')
#                print(f'epoch {epoch}: train accuracy - {max(max_acc_train)}')
                if self.validation_data is not None:
                    print(
                        f'epoch {epoch}: validation accuracy - {max([max(temp) for temp in self.validation_accuracies[epoch - eval_period:epoch]])}')
            if AUC > backup_AUC:
                self.best_model = copy.deepcopy(self.model)
                backup_AUC = AUC
            t2 = time.time()
            #print('times', t1-t0, t2-t1)
            self.accuracies.append(acc)
            self.losses.append(net_loss)
            self.steps.append(epoch)
            self.TP_rate.append(TP)
            self.FP_rate.append(FP)
#            self.train_accuracies.append(train_acc)
            net_loss = 0
        print(f'Finished training, best recorded model had AUC = {backup_AUC}')
        print(f'best group acc:  {max(self.group_acc)}   val acc: {max(self.group_val_acc)}')
        self.model = copy.deepcopy(self.best_model)

#    def group_eval(self):
#

    @staticmethod
    def name_counting_sort(name_arr, grade_arr):
        '''
        currently not a voting, just a summation
        '''
        unique_names = list(np.unique(name_arr))
        unique_names.sort()
        group_grades = np.zeros(np.shape(unique_names))
        real_grades = copy.deepcopy(grade_arr)
        real_grades[real_grades == 0] = -1
        for name, grade in zip(list(name_arr), grade_arr):
            group_grades[unique_names.index(name)] += grade-0.5
        grade_dict = {'pick_names': unique_names, 'group_decision': group_grades}
        return grade_dict

    def evaluate(self, threshold=0.5, test_set='test', current=True):
        """
        function to evaluate model performance
        @param threshold - determines threshold for classifying a pick as a success or failure
        @param test_set - determines if we check on training, testing or validation set
        @param current - Bool, determines if we use current model or best saved model
        """
        outputs = np.array([])
        test_labels = np.array([])

        if current:
            model_to_test = self.model
        else:
            model_to_test = self.best_model
        model_to_test.eval()
        last_ind = 1
        acc = 0
        count = 0
        last_ind_output = []
        last_ind_label = []
        final_indexes = []
        if test_set == 'test':
            data = self.test_data
            data_shape = self.test_size
        elif test_set == 'train':
            data = self.train_data
            data_shape = self.train_size
        elif test_set == 'validation':
            data = self.validation_data
            data_shape = self.validation_size

        # loads outputs and test labels into lists for evaluation
        all_names = []
        flag = True
        # x is either [episode_length,num_sensors] or [batch_size,episode_length,num_sensors]
        for x, y, lens, names in data:
            if flag:
                flag=False
                temp = np.shape(x)
                if len(temp) == 2:
                    reshape_param = temp[0]
                elif len(temp) == 3:
                    reshape_param = temp[0]*temp[1]
            hidden_layer = model_to_test.init_hidden(reshape_param)
            final_indexes.append(lens)
#            print('final indexes while we goin', final_indexes)
            x = torch.reshape(x, (reshape_param, 1, self.input_dim))
            y = torch.reshape(y, (reshape_param, last_ind))
            if self.model_type == 'LSTM':
                hidden_layer = tuple([e.data for e in hidden_layer])
            out, hidden_layer = model_to_test(x.to(self.device).float(), hidden_layer)
            count += 1
#            print(out.to('cpu').detach().numpy()[-1])
            outputs = np.append(outputs, out.to('cpu').detach().numpy())
            test_labels = np.append(test_labels, y.to('cpu').detach().numpy())
            all_names.extend(names[0])
            

        if self.eval_type == 'last':
            # Evaluates only the last point in the sequence
#            print('final indexes before doing the for loop,', final_indexes)
            final_indexes = [sum(final_indexes[:i]) - 1 for i in range(1, len(final_indexes) + 1)]
            # this only works since all examples are the same length
#            print('final indexes after the for loop',final_indexes)
            last_ind_output = outputs[final_indexes]
            last_ind_label = test_labels[final_indexes]
            num_pos = np.count_nonzero(last_ind_label)
            num_total = len(last_ind_label)
#            print('OUTPUT')
            group_grade = AppleClassifier.name_counting_sort(all_names, last_ind_output)
#            print('ACTUAL LABEL')
            correct_group_grade = AppleClassifier.name_counting_sort(all_names, last_ind_label)
            correct_group_grade['group_decision'] = correct_group_grade['group_decision'] > 0
            assert(len(group_grade['group_decision']) == len(correct_group_grade['group_decision']))
            num_group_total = len(group_grade['group_decision'])
            num_group_pos = np.count_nonzero(correct_group_grade['group_decision'])
            FP_group, TP_group, thresholds_group = metrics.roc_curve(correct_group_grade['group_decision'], group_grade['group_decision'])
            acc_group = (TP_group * num_group_pos + (1 - FP_group) * (num_group_total - num_group_pos)) / num_group_total
            AUC_group = metrics.roc_auc_score(correct_group_grade['group_decision'],group_grade['group_decision'])
            FP, TP, thresholds = metrics.roc_curve(last_ind_label, last_ind_output)
            acc = (TP * num_pos + (1 - FP) * (num_total - num_pos)) / num_total
            AUC = metrics.roc_auc_score(last_ind_label, last_ind_output)
#            print('for this round: ')
#            print('Group accuracy = ', max(acc_group))
#            print('Non-group accuracy = ', max(acc))
            if test_set == 'test':
                self.group_acc.append(max(acc_group))
            else:
                self.group_val_acc.append(max(acc_group))
        elif self.eval_type == 'alt_last':
            # Evaluates only the last point in the sequence with 3 classifications, success, failure and undecided
            # this only works since all examples are the same length
            final_indexes = list(range(data_shape[1] - 1, len(outputs), data_shape[1]))
            last_ind_output = outputs[final_indexes]
            last_ind_label = test_labels[final_indexes]
            confident_area = ((last_ind_output >= 0.75) | (last_ind_output <= 0.25))
            temp = np.array(last_ind_label[confident_area]) - (np.array(last_ind_output[confident_area]) > 0.5)
            acc = 1 - np.count_nonzero(temp) / len(final_indexes)
            FP = np.count_nonzero(temp < 0) / (len(last_ind_label) - np.count_nonzero(last_ind_label))
            TP = 1 - (np.count_nonzero(temp > 0) / np.count_nonzero(last_ind_label))
            AUC = metrics.roc_auc_score(last_ind_label, last_ind_output)
        elif self.eval_type == 'pick':
            # If 5 points in a row are the same classification, classify the pick as that, otherwise keep going
            final_indexes = [sum(final_indexes[:i]) - 1 for i in range(1, len(final_indexes) + 1)]
            # again, this will be nasty
            label_data = {'classification': [], 'timestep': []}
            for i in range(len(final_indexes) - 1):
                for j in range(final_indexes[i], final_indexes[i + 1] - 5):
                    timestep = final_indexes[i + 1] - final_indexes[i]
                    classification = True
                    if all(outputs[j:j + 5] < threshold):
                        classification = False
                        timestep = j + 5 - final_indexes[i]
                        break
                label_data['classification'].append(classification)
                label_data['timestep'].append(timestep)
            last_ind_label = test_labels[final_indexes[1:]]
            temp = np.array(last_ind_label) - np.array(label_data['classification'])
            acc = 1 - np.count_nonzero(temp) / len(last_ind_label)
            FP = np.count_nonzero(temp < 0) / (len(last_ind_label) - np.count_nonzero(last_ind_label))
            TP = 1 - (np.count_nonzero(temp > 0) / np.count_nonzero(last_ind_label))
            AUC = metrics.roc_auc_score(last_ind_label, last_ind_output)
            self.label_times.append(label_data)
        else:
            # Check every datapoint in the sequence
            num_pos = np.count_nonzero(outputs)
            num_total = len(outputs)
            FP, TP, thresholds = metrics.roc_curve(test_labels, outputs)
            acc = (TP * num_pos + (1 - FP) * (num_total - num_pos)) / num_total
            AUC = metrics.roc_auc_score(test_labels, outputs)
        model_to_test.train()

        return acc, TP, FP, AUC

    def evaluate_episode(self):
        """
        function to evaluate a single episode and return results for visualization
        @return normalized_feature - Z force for full sequence normalized to 0-1
        @return y - correct output
        @return outputs - model output
        @return name - name of pick being evaluated
        """
        self.model.eval()
        last_ind = 1
        plot_data = list(self.test_data)[self.plot_ind]
        x, y, name = plot_data[0], plot_data[1], plot_data[3]
        hidden_layer = self.model.init_hidden(np.shape(x)[0])
        x = torch.reshape(x, (np.shape(x)[0], 1, self.input_dim))
        y = torch.reshape(y, (np.shape(y)[0], last_ind))
        if self.model_type == 'LSTM':
            hidden_layer = tuple([e.data for e in hidden_layer])
        out, hidden_layer = self.model(x.to(self.device).float(), hidden_layer)
        outputs = out.to('cpu').detach().numpy()
        self.model.train()
        normalized_feature = x[:, :, 3]
        normalized_feature = (normalized_feature - min(normalized_feature)) / (
                max(normalized_feature) - min(normalized_feature))
        return normalized_feature, y, outputs, name

    def save_model(self, filename=None):
        """
        function to save model in models folder
        """
        if filename is None:
            filename = './models/' + self.identifier + \
                       datetime.datetime.now().strftime("%m_%d_%y_%H%M")
        torch.save(self.best_model, filename + '.pt')

    def save_data(self, filename=None):
        """
        function to save performance data in generated_data folder
        """
        if filename is None:
            filename = './generated_data/' + self.identifier + \
                       datetime.datetime.now().strftime("%m_%d_%y_%H%M")
        classifier_dict = {'acc': self.accuracies, 'loss': self.losses,
                           'steps': self.steps, 'TP': self.TP_rate,
                           'FP': self.FP_rate, 'ID': self.identifier,
                           'train_acc': self.train_accuracies, 'validation_acc': self.validation_accuracies}
        file = open(filename + '.pkl', 'wb')
        pkl.dump(classifier_dict, file)
        file.close()

        filename2 = './generated_data/' + self.identifier + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '_group_stats'
        group_dict = {'group acc': self.group_acc, 'group validation acc': self.group_val_acc}
        file = open(filename2 + '.pkl', 'wb')
        pkl.dump(group_dict, file)
        file.close()
        