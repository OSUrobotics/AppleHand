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
import csv

class AppleClassifier:
    def __init__(self, train_dataset, validation_dataset, param_dict, model=None, test_dataset=None):
        """
        A class for training, evaluating and generating plots of a classifier
        for apple pick data
        @param train_loader - RNNDataset with training data loaded
        @param validation_loader - RNNDataset with validationing data loaded
        @param param_dict - Dictionary with parameters for model, training, etc
        for detail on acceptable parameters, look at valid_parameters.txt
        """

        self.eval_type = 'last'
        
        # set hyperparameters based on parameter dict
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
            print('we are here ',validation_dataset.shape[2])
            self.input_dim = validation_dataset.shape[2]
        try:
            self.outputs = param_dict['outputs']
        except KeyError:
            self.outputs = "grasp"
        try:
            self.batch_size = param_dict['batch_size']
        except KeyError:
            self.batch_size = 1
        try:
            self.phase = param_dict['phase']
        except KeyError:
            self.phase = 'NA'
        self.output_dim = 1
        # set s/f sampling probability
        try:
            print('setting s/f balance to', param_dict['s_f_bal'])
            num_pos = 0
            num_neg = 0
            for state, label, lens, names in train_dataset:
                if min(label) <= 0:
                    num_neg += 1
                elif min(label) >= 0.5:
                    num_pos += 1
                else:
                    print('error! episode neither success or failure. setting weight to 0.5')
                    print(min(label))
            pos_ratio = 0.5
            neg_ratio = (1-param_dict['s_f_bal'])*num_pos*pos_ratio/(num_neg*param_dict['s_f_bal'])
        except TypeError:
            pos_ratio = 0.5
            neg_ratio = 0.5
            print('no desired s_f balance, sampling all episodes equally')
        s_f_bal = []
        for state, label, lens, names in train_dataset:
            if min(label) <= 0:
                s_f_bal.append(neg_ratio)
            elif min(label) >= 0.5:
                s_f_bal.append(pos_ratio)
            else:
                print('error! episode neither success or failure. setting weight to 0.5')
                print(min(label))
                s_f_bal.append(0.5)
        self.data_sampler = WeightedRandomSampler(s_f_bal, param_dict['batch_size'], replacement=False)
        
        self.train_data = DataLoader(train_dataset, shuffle=False,
                                     batch_size=param_dict['batch_size'], sampler=self.data_sampler)
        self.validation_data = DataLoader(validation_dataset, shuffle=False,
                                        batch_size=param_dict['batch_size'])
        
        if test_dataset is not None:
            self.test_data = DataLoader(test_dataset, shuffle=False,
                                              batch_size=param_dict['batch_size'])
            self.test_size = test_dataset.shape
            self.test_acc = []
            self.test_AUC = []
            self.group_val_acc = []
            self.group_val_AUC = []
        else:
            self.test_data = None
            self.test_size = None
            self.test_acc = []
            self.test_AUC = []
            self.group_val_acc = []
            self.group_val_AUC = []
        self.train_accuracies = []
        self.accuracies = []
        self.group_acc = []
        self.group_AUC = []
        self.AUC = []
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
                self.loss_fn = self.model.loss
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.best_model = copy.deepcopy(self.model)
        self.val_model = copy.deepcopy(self.model)
        self.plot_ind = 0
        self.validation_size = validation_dataset.shape
        self.train_size = train_dataset.shape
        self.identifier = self.outputs
        self.generate_ID()
        self.label_times = []
        self.metadata = []
        print('size of parameters = ', self.count_parameters())

    def count_parameters(self):
        """
        function to calculate number of trainable weights in model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_dataset(self, train_data, validation_data):
        """
        function to load train and validation data
        @params - must be RNNDatasets
        """
        self.train_data = DataLoader(train_data, shuffle=False,
                                     batch_size=None)
        self.validation_data = DataLoader(validation_data, shuffle=False,
                                    batch_size=None)
        self.validation_size = validation_data.shape
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
            self.test_acc = temp_dict['test_acc']
        except KeyError:
            pass

    def get_data_dict(self):
        """
        function to generate dict of performance during training for saving
        """
        classifier_dict = {'steps': self.steps, 'loss': self.losses,
                            'acc': self.accuracies, 'AUC': self.AUC,
                           'test_acc': self.test_acc, 'test_AUC': self.test_AUC,
                           'ID': self.identifier}
        return classifier_dict.copy()

    def get_best_performance(self, ind_range):
        """
        function to parse ROC response from scikit.metrics into max acc and best TP/FP over a window of epochs
        @param ind_range - list containing start and end epoch to consider
        """
        if (ind_range[0] >= 0) and (ind_range[1] < 0):
            ind_range[1] = len(self.accuracies) + ind_range[1]
        max_acc_ind = [np.argmax(a) for a in self.accuracies[ind_range[0]:ind_range[1]]]
        max_acc = []
        for i in range(ind_range[1] - ind_range[0]):
            max_acc.append(self.accuracies[ind_range[0] + i][max_acc_ind[i]])
        return max_acc

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
        eval_period = 2
        print('at start of train,', next(self.model.parameters()).is_cuda)
        if torch.cuda.is_available():
            self.model.cuda()
            print('cuda available')
        print('at end of train,', next(self.model.parameters()).is_cuda)
        #need to validation with differnt lr
        optim = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.model.train()
        print('starting training, finding the starting accuracy for random model of type', self.phase)
        accs, AUC, group_acc, group_AUC = self.evaluate(0.5)
#        train_acc, _, _, _ = self.evaluate(0.5, 'train') # currently we can't do train acc because the sampler fucks with the way we do eval
#        self.train_accuracies.append(train_acc) # until there is a need for it, i won't be fixing it

        if self.test_data is not None:
                test_accs, test_AUC, test_group_acc, test_group_AUC = self.evaluate(0.5, 'test')
                self.test_acc.append(test_accs)
                self.group_val_acc.append(test_group_acc)
                self.test_AUC.append(test_AUC)
                self.group_val_AUC.append(test_group_AUC)
        print(f'starting: AUC - {AUC}')
        self.accuracies.append(accs)
        self.AUC.append(AUC)
        self.group_acc.append(group_acc)
        self.group_AUC.append(group_acc)
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
                    hiddens = self.model.init_hidden(self.batch_size)
                    if self.model_type == "GRU":
                        hiddens = hiddens.data
                    else:
                        hiddens = tuple([e.data for e in hiddens])
                    pred, hiddens = self.model(x.to(self.device).float(), hiddens, lens)
                    for param in self.model.parameters():
                        if torch.isnan(param).any():
                            print('shit went sideways')
                    if self.eval_type == 'last':
                        loss = self.loss_fn(pred, label.to(self.device).float())
                    else:
                        loss = self.loss_fn(pred, label.to(self.device).float())
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    net_loss += float(loss)
                    epoch_loss += float(loss)
                    step += 1
            t1 = time.time()
            print(f'epoch {epoch} finished')
            accs, AUC, group_acc, group_AUC = self.evaluate(0.5)
#            train_acc, train_tp, train_fp, train_AUC = self.evaluate(0.5, 'train')
            if self.test_data is not None:
                test_accs, test_AUC, test_group_acc, test_group_AUC = self.evaluate(0.5, 'test')
                self.test_acc.append(test_accs)
                self.group_val_acc.append(test_group_acc)
                self.test_AUC.append(test_AUC)
                self.group_val_AUC.append(test_group_AUC)
                if test_AUC > val_AUC:
                    self.val_model = copy.deepcopy(self.model)
                    val_AUC = test_AUC
            if epoch % eval_period == 0:
                max_acc = self.get_best_performance([epoch - eval_period, epoch])
                best_epoch = np.argmax(max_acc)
                print(
                    f'epoch {epoch}: best validation accuracy  - {max_acc[best_epoch]}, net loss - {sum(self.losses[epoch - eval_period:epoch])}, Best AUC - {max(self.AUC[epoch - eval_period:epoch])}')
#                print(f'epoch {epoch}: train accuracy - {max(max_acc_train)}')
                if self.test_data is not None:
                    print(
                        f'epoch {epoch}: test accuracy - {max([max(temp) for temp in self.test_acc[epoch - eval_period:epoch]])} test AUC - {max(self.test_AUC[epoch - eval_period:epoch])}')
            if AUC > backup_AUC:
                self.best_model = copy.deepcopy(self.model)
                backup_AUC = AUC
            t2 = time.time()
            self.accuracies.append(accs)
            self.AUC.append(AUC)
            self.group_acc.append(group_acc)
            self.group_AUC.append(group_acc)
            self.losses.append(net_loss)
            self.steps.append(epoch)
#            self.train_accuracies.append(train_acc)
            net_loss = 0
        temp_list = []
        for acc_list in self.accuracies:
            temp_list.append(np.max(acc_list))
        acc_max = np.max(temp_list)
        temp_list = []
        
        self.metadata = [backup_AUC, acc_max]
        print(f'Finished training, best recorded model had proxy AUC = {backup_AUC}')
        print(f'Finished training, best recorded model had proxy ACC = {acc_max}')
        if self.test_data is not None:
            for acc_list in self.test_acc:
                temp_list.append(np.max(acc_list))
            val_max = np.max(temp_list)
            print(f'Finished training, best recorded model had real AUC = {val_AUC}')
            print(f'Finished training, best recorded model had real ACC = {val_max}')
            self.metadata.extend([val_AUC, val_max])
        self.model = copy.deepcopy(self.best_model)

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

    def evaluate(self, threshold=0.5, test_set='validation', current=True, ROC=False):
        """
        function to evaluate model performance
        @param threshold - determines threshold for classifying a pick as a success or failure
        @param test_set - determines if we check on training, testing or test set
        @param current - Bool, determines if we use current model or best saved model
        """
#        print('at start of eval',next(self.model.parameters()).is_cuda)
        test_labels = np.array([])
#        if torch.cuda.is_available():
#            self.model.cuda()
#            print('in eval, cuda available')
        if current:
            model_to_test = self.model
        else:
            if torch.cuda.is_available():
                self.best_model.cuda()
                print('cuda available')
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
        outputs = np.array([[]])
        # x has shape [batch_size,episode_length,num_sensors]
        for x, y, lens, names in data:
#            print('lens')
#            input(lens)
            end_output_shape = np.shape(y)
            hidden_layer = model_to_test.init_hidden(self.batch_size)
#            print(hidden_layer.device.type)
            final_indexes.extend(lens.tolist())
            if self.model_type == 'LSTM':
                hidden_layer = tuple([e.data for e in hidden_layer])
#            print(hidden_layer.device.type)
            out, hidden_layer = model_to_test(x.to(self.device).float(), hidden_layer,lens)
            count += 1
            start_out_shape = out.shape
            temp = np.ones(end_output_shape) * 2
            out = out.to('cpu').detach().numpy()
            y = y.to('cpu').detach().numpy()
            # input(y[0])
            temp[:,:start_out_shape[-1]] = out
            if flag:
                outputs = temp.copy()
                test_labels = y.copy()
                flag=False
            else:
                outputs = np.append(outputs, temp, axis=0)
                test_labels = np.append(test_labels,y, axis=0)
            all_names.extend(list(names[0][0]))

        if self.eval_type == 'last':
            # Evaluates only the last point in the sequence
            last_ind_output = []
            last_ind_label = []
            for i, ind in enumerate(final_indexes):
                last_ind_output.append(outputs[i,ind-1])
                last_ind_label.append(test_labels[i,ind-1])
            num_pos = np.count_nonzero(last_ind_label)
            num_total = len(last_ind_label)
            group_grade = AppleClassifier.name_counting_sort(all_names, last_ind_output)
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

        elif self.eval_type == 'pick':
            # If (Eval num) points in a row are the same classification, classify the pick as that, otherwise keep going
            eval_num = 5
            label_data = {'classification': [], 'timestep': []}
            for i, episode in enumerate(outputs):
                for j in range(len(episode)):
                    classification = True
                    if j > final_indexes[i]:
                        break
                    elif all(outputs[j:j + 5] < threshold):
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
        if ROC:
            return TP, FP, TP_group, FP_group, acc, acc_group
        else:
            return acc, AUC, acc_group, AUC_group

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
        plot_data = list(self.validation_data)[self.plot_ind]
        x, y, lens, name = plot_data[0], plot_data[1], plot_data[2], plot_data[3]
        hidden_layer = self.model.init_hidden(np.shape(x)[0])
        print(lens)
##        lens = torch.tensor(lens,dtype=int)
        print(x.shape, np.shape(x))
##        x = torch.reshape(x, (np.shape(x)[0], 1, self.input_dim))
##        y = torch.reshape(y, (np.shape(y)[0], last_ind))
#        print(x.shape)
#        print(y.shape)
#        print(lens)
        if type(lens) is not torch.Tensor:
            lens = torch.tensor(lens,dtype=int)
        if self.model_type == 'LSTM':
            hidden_layer = tuple([e.data for e in hidden_layer])
        out, hidden_layer = self.model(x.to(self.device).float(), hidden_layer, lens)
        outputs = out.to('cpu').detach().numpy()
        self.model.train()
        normalized_feature = x[:, :, 3]
        print(normalized_feature)
        print(normalized_feature)
        normalized_feature = (normalized_feature - normalized_feature.min()) / (
                normalized_feature.max() - normalized_feature.min())
        print(normalized_feature.shape)
        print(y.shape)
        print(outputs.shape)
        self.plot_ind +=1
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

        classifier_dict = self.get_data_dict()
        file = open(filename + '.pkl', 'wb')
        pkl.dump(classifier_dict, file)
        file.close()
        
    def save_metadata(self):
        filename = './test_records.csv'
        row = [self.epochs, self.layers, self.hidden, self.batch_size, self.phase]
        row.extend(self.metadata)
        with open(filename,'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            writer.writerow(row)