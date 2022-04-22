#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:20:43 2022

@author: orochi
"""

#grasp_epochs=1000_input_size=33_hidden_layers=202_07_22_1224
# Purpose of this file is to run an episode with a good classifier, process the lstm outputs into a 
# timebased rosbag and name it the name of the episode it is doing

import rospy
import numpy as np
import pickle as pkl
from pick_classifier_main import ExperimentHandler
from std_msgs.msg import Float32

handler = ExperimentHandler()
#handler.args.policy = 'grasp_epochs=1000_input_size=33_hidden_layers=202_07_22_1224'

handler.args.policy = 'grasp_epochs=100_input_size=3302_24_22_1525'
handler.run_experiment()

e_name, LSTM_out, label_out = handler.plot_example()

print('name of this episode is ', e_name)
print(f'there are {len(LSTM_out)} timesamples in this episode')
print(f'the full outputs should be as follows: {LSTM_out}')
grasp_time = input('input the time in seconds that a grasp takes')
pick_time = input('input the time in seconds that a pick takes')
#grasp_time = 2
#pick_time = 2.16
grasp_time = float(grasp_time)
pick_time = float(pick_time)
num_points = len(LSTM_out)

rospy.init_node('data_publisher')
lstmpub = rospy.Publisher('/lstm_output', Float32, queue_size=10)
labelpub = rospy.Publisher('/label_output', Float32, queue_size=10)

rospy.sleep(2)
#time_of_full = float(time_of_full)
count = 0
for lstm_val, label_val in zip(LSTM_out, label_out):
    lstmpub.publish(Float32(lstm_val))
    labelpub.publish(Float32(label_val))
    if count >= num_points/2:
        rospy.sleep(pick_time/num_points*2)
    else:
        rospy.sleep(grasp_time/num_points*2)