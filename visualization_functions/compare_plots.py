#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:47:22 2022

@author: orochi
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_epochs=1000_input_size=33_hidden_layers=202_08_22_1043.pkl','rb') as file:
#    set1 = pkl.load(file)
#
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_epochs=1000_input_size=33_hidden_layers=202_08_22_1055.pkl','rb') as file:
#    set2 = pkl.load(file)
#
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_epochs=1000_input_size=33_hidden_layers=202_08_22_1108.pkl','rb') as file:
#    set3 = pkl.load(file)
#
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_epochs=1000_input_size=33_hidden_layers=202_07_22_1224.pkl','rb') as file:
#    with_proxy = pkl.load(file)
#
#
## acc plot
#plt.plot(with_proxy['steps'],with_proxy['acc'])
#plt.plot(set1['steps'],set1['acc'])
#plt.plot(set2['steps'],set2['acc'])
#plt.plot(set3['steps'],set3['acc'])
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend(['Train on Proxy','Real World Set 1','Real World Set 2','Real World Set 3'])
#plt.show()
## tp/fp plot
#plt.plot(with_proxy['steps'],with_proxy['TP'])
#plt.plot(with_proxy['steps'],with_proxy['FP'])
#plt.plot(set1['steps'],set1['TP'])
#plt.plot(set1['steps'],set1['FP'])
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend(['Train on Proxy: TP','Train on Proxy: FP','Real World Set 1: TP','Real World Set 1: FP'])
#plt.show()
# ROC plots?


#    grasp_ablation_dict = {'num inputs': sizes, 'best accuracy': best_accuracies,
##                           'std dev': best_acc_std_dev, 'names': worst_names}
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data02_11_22_1713.pkl','rb') as file:
#    ablation_grasp = pkl.load(file)
#
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data02_18_22_2227.pkl','rb') as file:
#    ablation_pick = pkl.load(file)
#
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data02_16_22_1626.pkl','rb') as file:
#    ablation_both = pkl.load(file)

with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data/grasp_ablation_data06_11_22_2134.pkl','rb') as file:
    full_data_grasp = pkl.load(file)

with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data06_14_22_0718.pkl','rb') as file:
    full_data = pkl.load(file)
    
with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data06_11_22_2353.pkl','rb') as file:
    full_data_pick = pkl.load(file)

print([key for key in full_data_grasp.keys()])
full_data_grasp
#print(ablation_both)
print('full')
print(full_data['names'])
print('pick')
print(full_data_pick['names'])
print('grasp')
print(full_data_grasp['names'])

all_names = {'Arm Force X': [0],
              'Arm Force Y': [1],
              'Arm Force Z': [2],
              'Arm Torque Roll': [3],
              'Arm Torque Pitch': [4],
              'Arm Torque Yaw': [5],
              'IMU Acceleration X':[6, 15, 24],
              'IMU Acceleration Y':[7, 16, 25],
              'IMU Acceleration Z':[8, 17, 26],
              'IMU Gyro X': [9, 18, 27],
              'IMU Gyro Y': [10, 19, 28],
              'IMU Gyro Z': [11, 20, 29],
              'Finger Position': [12, 21, 30],
              'Finger Speed': [13, 22, 31],
              'Finger Effort': [14, 23, 32]}
all_names = [name for name in all_names.keys()]

for name in all_names:
    if name not in full_data['names']:
        print('in full phase, ',name,' is the best sensor')
    if name not in full_data_pick['names']:
        print('in pick phase, ',name,' is the best sensor')
    if name not in full_data_grasp['names']:
        print('in grasp phase, ',name,' is the best sensor')

## acc
#plt.errorbar(full_data_grasp['num inputs'], full_data_grasp['best accuracy'], full_data_grasp['accuracy std dev'], capsize=3, color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),linestyle='-')
#plt.errorbar(full_data_pick['num inputs'], full_data_pick['best accuracy'], full_data_pick['accuracy std dev'], capsize=3, color= (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), linestyle=':')
#plt.errorbar(full_data['num inputs'], full_data['best accuracy'], full_data['accuracy std dev'], capsize=3, color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),linestyle='-')


# AUC
plt.errorbar(full_data_grasp['num inputs'], full_data_grasp['best auc'], full_data_grasp['auc std dev'], capsize=3, color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),linestyle='-')
plt.errorbar(full_data_pick['num inputs'], full_data_pick['best auc'], full_data_pick['auc std dev'], capsize=3, color= (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), linestyle=':')
plt.errorbar(full_data['num inputs'], full_data['best auc'], full_data['auc std dev'], capsize=3, color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),linestyle='-')



#print(full_data_pick['names'])
#plt.errorbar(ablation_grasp['num inputs'], ablation_grasp['best accuracy'], ablation_grasp['std dev'], capsize=3, color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),linestyle='-')
#plt.errorbar(ablation_pick['num inputs'], ablation_pick['best accuracy'], ablation_pick['std dev'], capsize=3, color= (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),linestyle='--')
#plt.errorbar(ablation_both['num inputs'], ablation_both['best accuracy'], ablation_both['std dev'], capsize=3, color= (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), linestyle=':')
plt.legend(['Grasp Phase', 'Pick Phase','Both Grasp and Pick Phase'])
plt.xlabel('Number of Network Inputs')
plt.ylabel('AUC')
plt.show()


#also need to get the image of a single episode so that they know where the labels go and how that stuff works

#also need the rosbag thing