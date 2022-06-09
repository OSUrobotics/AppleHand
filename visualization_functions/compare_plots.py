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

with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data06_08_22_2029.pkl','rb') as file:
    full_data_grasp = pkl.load(file)

with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data06_08_22_1422.pkl','rb') as file:
    full_data = pkl.load(file)
    
#with open('/home/orochi/apple_picking/IROS_apple/AppleHand/generated_data/grasp_ablation_data03_12_22_0009.pkl','rb') as file:
#    full_data_pick = pkl.load(file)

#print(ablation_both)
plt.errorbar(full_data_grasp['num inputs'], full_data_grasp['best accuracy'], full_data_grasp['std dev'], capsize=3, color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),linestyle='-')
#plt.errorbar(full_data_pick['num inputs'], full_data_pick['best accuracy'], full_data_pick['std dev'], capsize=3, color= (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), linestyle=':')
plt.errorbar(full_data['num inputs'], full_data['best accuracy'], full_data['std dev'], capsize=3, color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),linestyle='-')
#print(full_data_pick['names'])
#plt.errorbar(ablation_grasp['num inputs'], ablation_grasp['best accuracy'], ablation_grasp['std dev'], capsize=3, color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),linestyle='-')
#plt.errorbar(ablation_pick['num inputs'], ablation_pick['best accuracy'], ablation_pick['std dev'], capsize=3, color= (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),linestyle='--')
#plt.errorbar(ablation_both['num inputs'], ablation_both['best accuracy'], ablation_both['std dev'], capsize=3, color= (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), linestyle=':')
plt.legend(['Grasp Phase', 'Both Grasp and Pick Phase'])
plt.xlabel('Number of Network Inputs')
plt.ylabel('Accuracy')
plt.show()


#also need to get the image of a single episode so that they know where the labels go and how that stuff works

#also need the rosbag thing