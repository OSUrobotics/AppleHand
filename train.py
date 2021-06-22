#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 23:25:48 2021

@author: orochi
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from RNNs import GRUNet, LSTMNet
import pickle as pkl
import datetime
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as tr

# Define data root directory


def flatten_arbitray(long_arr):
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = "/Deep_Learning_Data/"
print(os.listdir(dir_path+data_dir))
temp = os.listdir(dir_path+data_dir)
indicies = ['Batch' in temp[i] for i in range(len(temp))]
temp = np.array(temp)
names = temp[indicies]
states = []
labels = []
quats = []
for i in range(len(names)):
    print(data_dir+names[i])
    a=pkl.load(open(dir_path+data_dir + names[i], 'rb'))
    s = a['states'].tolist() # this needs fixin
    l = a['labels'].to_list()
    q = a['quats'].tolist() # this too
    if i == 0:
        states = np.copy(s)
        labels = np.copy(l)
        quats = np.copy(q)
    else:
        states=np.append(states, s,axis=1)
        labels=np.append(labels,l)
        quats=np.append(quats,q,axis=1)

states = np.transpose(states)
quats = np.transpose(quats)
label_scalers = {}
prev_labels = -1
episode_switches=[]
episodes = []
eventual_labels = []
eventual_quats = []
one_hot_labels = []
count = 0
#print(quats[0]==[-1,-1,-1,-1])
for i in range(len(labels)):
#    if all(quats[i] == -1):
#        count +=1
#    el
    if labels[i] == 0:
        #one_hot_labels.append([1,0,0,0])
        eventual_labels.append(0)
        eventual_quats.append(quats[i])
    elif labels[i] == 1:
        #one_hot_labels.append([0,1,0,0])
        eventual_labels.append(1)
        eventual_quats.append(quats[i])
    elif labels[i] == 2:
        #one_hot_labels.append([0,0,1,0])
        eventual_labels.append(2)
        eventual_quats.append(quats[i])
    elif labels[i] == 3:
        #one_hot_labels.append([0,0,0,1])
        eventual_labels.append(3)
        eventual_quats.append(quats[i])
    elif labels[i] == -1:
        count += 1
        eventual_labels.append(0)
#        #one_hot_labels.append([1,0,0,0])
#
#
    if (labels[i]==0) & (prev_labels!=0):
        episode_switches.append(i-count)
    prev_labels = labels[i]

for i in range(len(episode_switches)-1):
    episodes.append([episode_switches[i],episode_switches[i+1]])
np.random.shuffle(episodes)
# Split data into train/test portions and combining all data from different files into a single array
# I also need to keep the episodes in mind however. The order of shit matters
test_portion = int(0.1*len(episodes))
train_state = [states[episodes[i][0]:episodes[i][1]] for i in range(len(episodes)-test_portion)]
#train_label = [one_hot_labels[episodes[i][0]:episodes[i][1]] for i in range(len(episodes)-test_portion)]
train_label = [eventual_labels[episodes[i][0]:episodes[i][1]] for i in range(len(episodes)-test_portion)]


train_quat = [eventual_quats[episodes[i][0]:episodes[i][1]] for i in range(len(episodes)-test_portion)]
test_state = [states[episodes[-i-1][0]:episodes[-i-1][1]] for i in range(test_portion)]
#test_label = [one_hot_labels[episodes[-i-1][0]:episodes[-i-1][1]] for i in range(test_portion)]
test_label = [eventual_labels[episodes[-i-1][0]:episodes[-i-1][1]] for i in range(test_portion)]


test_quat = [eventual_quats[episodes[-i-1][0]:episodes[-i-1][1]] for i in range(test_portion)]
temp_quat=[]
temp_state=[]
temp_T_quat=[]
temp_T_state=[]

train_label=np.array(flatten_arbitray(train_label))
test_label=np.array(flatten_arbitray(test_label))
test_state=np.array(flatten_arbitray(test_state))
test_quat=np.array(flatten_arbitray(test_quat))
train_state=np.array(flatten_arbitray(train_state))
train_quat=np.array(flatten_arbitray(train_quat))


for i in range(len(train_label)):
    if train_label[i] ==1 :
        temp_quat.append(train_quat[i])
        temp_state.append(train_state[i])
for i in range(len(test_label)):
    if test_label[i] ==1 :
        temp_T_quat.append(test_quat[i])
        temp_T_state.append(test_state[i])

train_stem_state=np.array(temp_state)
test_stem_state=np.array(temp_T_state)
train_quat=np.array(temp_quat)
test_quat=np.array(temp_T_quat)

#print(train_quat)
batch_size = 5000


#thought 1. train two completely different RNNs
train_data_grasp = TensorDataset(torch.from_numpy(train_state), torch.from_numpy(train_label))
train_data_stem = TensorDataset(torch.from_numpy(train_stem_state), torch.from_numpy(train_quat))
train_loader_grasp = DataLoader(train_data_grasp, shuffle=False, batch_size=batch_size, drop_last=True)
train_loader_stem = DataLoader(train_data_stem, shuffle=False, batch_size=batch_size, drop_last=True)

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#input_dim, hidden_dim, output_dim, n_layers
def train(train_loader,test_data,test_label,epochs = 5,model_type='GRU',layers=4,hidden=100,output='grasp',drop_prob=0.2):
    # Set hyperparameters
    input_dim = 3
    if output == 'grasp':
        output_dim = 4
        loss_fn = nn.CrossEntropyLoss()
        network_type='g'
    elif output == 'stem':
        output_dim = 4
        loss_fn = quaternion_loss
        network_type='q'
    elif output == 'stem_aa':
        output_dim = 3
        loss_fn = axis_angle_loss
        network_type='aa'
    batch_size = 5000
    # Instantiate the models
    if model_type == 'GRU':
        model = GRUNet(input_dim,hidden,output_dim,layers,drop_prob)
    elif model_type == 'LSTM':
        model = LSTMNet(input_dim,hidden,output_dim,layers,drop_prob)
    # Define loss function and optimizer
    if torch.cuda.is_available():
        model.cuda()

    optim = torch.optim.Adam(model.parameters(),lr=0.0001)
    model.train()
    accs=[]
    losses=[]
    steps=[]
    acc = evaluate(model,test_data,test_label,network_type)
    #acc=0
    print(f'starting accuracy - {acc}')
    accs.append(acc)
    losses.append(0)
    steps.append(0)
    net_loss = 0
    for epoch in range(1,epochs+1):
        hiddens = model.init_hidden(batch_size)
        net_loss = 0
        epoch_loss = 0
        step = 0
        print(f'epoch {epoch} start')
        for x, label in train_loader:
            #print(x.shape,label.shape)
            x=torch.reshape(x,(5000,1,3))
            #print(x.shape,label.shape)
            #label=torch.unsqueeze(label,0)
            #x=torch.unsqueeze(x,0)
            if model_type == "GRU":
                hiddens = hiddens.data
            else:
                hiddens = tuple([e.data for e in hiddens])
            #print(x.shape)
            pred, hiddens = model(x.to(device).float(),hiddens)
            for param in model.parameters():
                if torch.isnan(param).any():
                    print('shit went sideways')
            #print('label',label[0])
            #print('prediction',torch.max(pred),torch.min(pred))
            if network_type == 'g':
                loss = loss_fn(pred,label.to(device).long())
            else:
                loss = loss_fn(pred.to('cpu'),label.to('cpu').float())
            optim.zero_grad()
            #print(loss)
            loss.backward()
            optim.step()

            net_loss += loss
            epoch_loss += loss
#            if step % 100 == 0:
#                acc = evaluate(model,test_data,test_label,grasp_flag)
#                print(f'step {step}: accuracy - {acc}, loss - {net_loss}')
#                accs.append(acc)
#                losses.append(net_loss)
#                steps.append(epoch)
#                net_loss = 0
#            step += 1
        #for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            #print('gradient norms',p.grad.data.norm(2).item())
        acc = evaluate(model,test_data,test_label,network_type)
        print(f'epoch {epoch}: accuracy - {acc}, loss - {epoch_loss}')
        accs.append(acc)
        losses.append(net_loss)
        steps.append(epoch)
        net_loss = 0
#        np.random.shuffle(episodes)
#        train_state = [states[episodes[i][0]:episodes[i][1]] for i in range(len(episodes)-test_portion)]
#        train_label = [labels[episodes[i][0]:episodes[i][1]] for i in range(len(episodes)-test_portion)]
        #print(f'epoch loss: {epoch_loss}')
    return model, accs, losses, steps

def quaternion_difference(q1,q2):
    if len(q1) == 3:
        q1 = tr.matrix_to_quaternion(tr.axis_angle_to_matrix(torch.tensor(q1)))
    #q1 is from network, q2 is from label
    #q1 is [real,i,j,k]
    #q2 is [i,j,k,real]
    q2t = [-q2[0],-q2[1],-q2[2],q2[3]]
    q1t = [q1[1],q1[2],q1[3],q1[0]]
    q1t = R.from_quat(q1t)
    q2t = R.from_quat(q2t)
    #diff parts are now both [i, j, k,real]
    diff = (q1t*q2t).as_quat()
    diff = diff/np.linalg.norm(diff)
    ang = 2*np.arccos(abs(diff[3]))
    return ang

def axis_angle_loss(axis_angle,quat):
    output = tr.axis_angle_to_matrix(axis_angle)
    output = torch.transpose(output,1,2)
    q2c = torch.zeros(quat.shape)
    q2c[:,0] = quat[:,3]
    q2c[:,1:] = quat[:,0:3]
    label= tr.quaternion_to_matrix(q2c)
    diff = torch.acos((torch.diagonal(torch.matmul(output,label), dim1=-2, dim2=-1).sum(-1)-1)/2)
    return diff.mean()
#
#def quaternion_diff(q1,q2):
#    #q1 is from network, q2 is from label
#    #q1 is [real,i,j,k]
#    #q2 is [i,j,k,real]
#    q2c = torch.zeros(q2.shape)
#    q2c[:,0] = q2[:,3]
#    q2c[:,1:] = -q2[:,0:3]
#    #diff parts are now both [real, i, j, k]
#    diff = tr.quaternion_multiply(q1,q2c)
#    diff = nn.functional.normalize(diff)
#    ang = 2*torch.acos(abs(diff[:,0]))
#    return ang.sum()

def quaternion_loss(q1,q2):
    #q1 is from network, q2 is from label
    #q1 is [real,i,j,k]
    #q2 is [i,j,k,real]
    q2c = torch.zeros(q2.shape)
    q2c[:,0] = q2[:,3]
    q2c[:,1:] = -q2[:,0:3]
    #diff parts are now both [real, i, j, k]
    diff = tr.quaternion_multiply(q1,q2c)
    diff = nn.functional.normalize(diff)
    ang = 2*torch.acos(abs(diff[:,0]))
    return ang.mean()

def evaluate(model,test_data,test_labels,network_type):
    model.eval()
    hidden_layer = model.init_hidden(1)
    if network_type == 'aa':
        outputs=torch.zeros((len(test_labels),3))
    else:
        outputs=torch.zeros((len(test_labels),4))

    acc = 0
    count = 0
    for x,y in zip(test_data,test_labels):
        
        x=torch.tensor(x)
        x=torch.unsqueeze(x,0)
        x=torch.unsqueeze(x,0)
        out, hidden_layer = model(x.to(device).float(),hidden_layer)
        #print(out,outputs[0])
        outputs[count]=out
        count+=1

    #if not grasp_flag:
    #    acc = quaternion_loss(outputs.to('cpu'),torch.tensor(test_labels))
    #else:
    for i in range(len(outputs)):
        #grasp acc
        if network_type == 'g':
            temp = abs(np.argmax(outputs[i].to('cpu').detach().numpy()) - test_labels[i])
            if temp >= 1:
                acc += 0
            else:
                acc += 1
        else: # 
            #acc += sum(abs(outputs[i].to('cpu').detach().numpy()[0] - test_labels[i]))
            temp = abs(quaternion_difference(outputs[i].to('cpu').detach().numpy(),test_labels[i]))
            acc += min(temp,abs(temp-2*3.14159))
    acc = acc/len(outputs)
    model.train()
    return acc


def evaluate_secondary(model,test_data,test_labels):
    model.eval()
    hidden_layer = model.init_hidden(1)
    outputs=[]
    acc = 0
    for x,y in zip(test_data,test_labels):
        x=torch.tensor(x)
        x=torch.unsqueeze(x,0)
        x=torch.unsqueeze(x,0)
        out, hidden_layer = model(x.to(device).float(),hidden_layer)
        outputs.append(out)
    model.train()
    return outputs


#plt.scatter(range(len(train_quat[:,0])),train_quat[:,0])
#plt.scatter(range(len(train_quat[:,0])),train_quat[:,1])
#plt.scatter(range(len(train_quat[:,0])),train_quat[:,2])
#plt.scatter(range(len(train_quat[:,0])),train_quat[:,3])
#plt.show()
#input('look right?')
flag = False

while not flag:
    trained_stem_gru, stem_accuracies, stem_losses, stem_steps = train(train_loader_stem,test_state,test_quat,epochs=25,output='stem_aa')
    print('model 1 finished, saving now')
    torch.save(trained_stem_gru,'stem_gru_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
    stem_gru_dict={'acc':stem_accuracies, 'loss': stem_losses, 'steps': stem_steps}
    file = open('stem_gru_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
    pkl.dump(stem_gru_dict,file)
    file.close()
    flag = True


flag = False

while not flag:
    try:
        trained_stem_gru, gru_accuracies, gru_losses, gru_steps = train(train_loader_stem,test_state,test_quat,epochs=25,output='stem')
        print('model 1 finished, saving now')
        torch.save(trained_stem_gru,'stem_gru_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
        stem_gru_dict={'acc':stem_accuracies, 'loss': stem_losses, 'steps': stem_steps}
        file = open('stem_gru_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
        pkl.dump(stem_gru_dict,file)
        file.close()
        flag = True
    except:
        print('stem gru failed')
#while not flag:
#    try:
#        trained_grasp_gru, grasp_accuracies, grasp_losses, grasp_steps = train(train_loader_grasp,test_state,test_label,epochs=10,drop_prob=0.2)
#        print('model 2 finished, saving now')
#        torch.save(trained_grasp_gru,'grasp_gru_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
#        grasp_gru_dict={'acc':grasp_accuracies, 'loss': grasp_losses, 'steps': grasp_steps}
#        file = open('grasp_gru_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
#        pkl.dump(grasp_gru_dict,file)
#        file.close()
#        flag = True
#    except:
#        print('grasp gru failed')
#
'''
drop_prods=[0.0002,0.002,0.02,0.2]
drop_accs=[]
for i in range(4):
    flag = False
    while not flag:
        try:
            trained_grasp_gru, grasp_accuracies, grasp_losses, grasp_steps = train(train_loader_grasp,test_state,test_label,epochs=25,drop_prob=drop_prods[i])
            print('model', i,'finished, saving now')
            drop_accs.append(grasp_accuracies)
            torch.save(trained_grasp_gru,'grasp_gru_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
            grasp_gru_dict={'acc':grasp_accuracies, 'loss': grasp_losses, 'steps': grasp_steps}
            file = open('grasp_gru_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
            pkl.dump(grasp_gru_dict,file)
            file.close()
            flag = True
        except:
            print('grasp gru failed')
            
layers = [1,2,4,8]
layer_accs = []
for i in range(4):
    flag = False
    while not flag:
        try:
            trained_grasp_gru, grasp_accuracies, grasp_losses, grasp_steps = train(train_loader_grasp,test_state,test_label,epochs=25,layers=layers[i])
            print('model',i,'finished, saving now')
            layer_accs.append(grasp_accuracies)
            torch.save(trained_grasp_gru,'grasp_gru_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
            grasp_gru_dict={'acc':grasp_accuracies, 'loss': grasp_losses, 'steps': grasp_steps}
            file = open('grasp_gru_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
            pkl.dump(grasp_gru_dict,file)
            file.close()
            flag = True
        except:
            print('grasp gru failed')



flag = False
while not flag:
    try:
        trained_grasp_lstm, accuracies, losses, steps = train(train_loader_grasp,test_state,test_label,epochs=25,model_type='LSTM')
        print('model 3 finished, saving now')
        torch.save(trained_grasp_lstm,'grasp_lstm_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
        grasp_lstm_dict={'acc': accuracies, 'loss': losses, 'steps': steps}
        file = open('grasp_lstm_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
        pkl.dump(grasp_lstm_dict,file)
        file.close()
        flag = True
    except:
        print('grasp lstm failed')


flag = False
while not flag:
    try:
        trained_stem_lstm, lstm_accuracies, lstm_losses, lstm_steps = train(train_loader_stem,test_state,test_quat,epochs=25,model_type='LSTM',output='stem')
        print('model 4 finished, saving now')
        torch.save(trained_stem_lstm,'stem_lstm_'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pt')
        stem_lstm_dict={'acc':lstm_accuracies, 'loss': lstm_losses, 'steps': lstm_steps}
        file = open('stem_lstm_data'+datetime.datetime.now().strftime("%m_%d_%y_%H%M")+'.pkl','wb')
        pkl.dump(stem_lstm_dict,file)
        file.close()
        flag = True
    except:
        print('stem lstm failed')
'''
#plt.plot(np.array(steps),accuracies)
#plt.plot(np.array(grasp_steps),grasp_accuracies)
#plt.legend(['LSTM','GRU'])
#plt.xlabel('Steps')
#plt.ylabel('Accuracy')
#plt.show()
#
plt.plot(np.array(gru_steps),gru_accuracies)
plt.plot(np.array(stem_steps),stem_accuracies)
plt.legend(['Quaternion Loss','Axis Angle Loss'])
plt.xlabel('Epochs')
plt.ylabel('Average Rotation Angle Error')
plt.show()

#gru_outputs=evaluate_secondary(trained_grasp_gru,test_state,test_label)
#for i in range(len(gru_outputs)):
#    gru_outputs[i] = gru_outputs[i].to('cpu').detach().numpy()[0]
#    gru_outputs[i] = np.argmax(gru_outputs[i])
#plt.scatter(range(len(test_label)),test_label)
#plt.scatter(range(len(gru_outputs)),gru_outputs)
#plt.legend(['correct labels','gru labels'])
#plt.show()