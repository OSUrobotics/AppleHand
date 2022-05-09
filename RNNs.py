#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 23:29:29 2021

@author:
"""
import torch
import torch.nn as nn

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, network_type = 'aa'):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Reduce complexity of inputs to RNN
        RNN_in = 5
        self.fl = nn.Linear(input_dim, RNN_in)
        self.network_type = network_type
        self.gru = nn.GRU(RNN_in, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x, h):
        self.gru.flatten_parameters()
        rnn_x = self.fl(x)
        out, h = self.gru(rnn_x, h)
        out = self.fc(self.relu(out[:,-1]))
        out = self.sig(out)
        if self.network_type == 'q':
            out = self.tanh(out)
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2,grasp_flag = True):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Reduce complexity of inputs to RNN
        RNN_in = input_dim#5
        self.fl = nn.Linear(input_dim, RNN_in)
        self.grasp_flag = grasp_flag
        self.lstm = nn.LSTM(RNN_in, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self, x, h, xlen):
        self.lstm.flatten_parameters()
        
#        rnn_x = self.fl(x)
        rnn_x = nn.utils.rnn.pack_padded_sequence(x, xlen, batch_first=True, enforce_sorted=False)
        out, h = self.lstm(rnn_x, h)
#        print('out right after lstm',out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
#        print('out shape after padding again',out.shape)
#        input(out[:,:,-1].shape)
#        out = self.fc(self.relu(out[:,:,-1]))
#        
        if not self.grasp_flag:
            out = self.tanh(out)
        else:
            out = self.sig(out[:,:,-1])
        
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
    
    def loss(self, output, labels):
        loss_fn = nn.MSELoss(reduction='none')
        tag_pad_token = 2
        mask = (labels != tag_pad_token).float()
        output = output * mask[:,:output.shape[-1]]
        full_loss = loss_fn(output,labels[:,:output.shape[-1]]).mean()
        return full_loss