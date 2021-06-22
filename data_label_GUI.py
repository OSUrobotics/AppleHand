import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import bagpy
import pandas as pd
from bagpy import bagreader
import math
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import pickle as pkl
import datetime

#sns.set() # Setting seaborn as default style even if use only matplotlib

# -------------------------- Step 1: Open Bagfiles ---------------------------

# Open the bag file with the bagreader
#file[0] = 'bagfiles/Exp1_1_(10noise_top_10gripper)_2021-05-06-19-01-53.bag'
#b = bagreader(file[0])
#exp = 11

#b = bagreader('bagfiles/Exp1_2_(20noise_top_10gripper)_2021-05-06-19-06-08.bag')
#exp = 12

#b = bagreader('bagfiles/Exp1_3_(30noise_top_10gripper)_2021-05-06-19-09-49.bag')
#exp = 13

#b = bagreader('bagfiles/Exp2_1_(10noise_45_10gripper)_2021-05-06-20-30-48.bag')
#exp = 21

#b = bagreader('bagfiles/Exp2_2_(20noise_45_10gripper)_2021-05-06-20-26-20.bag')
#exp = 22

#b = bagreader('bagfiles/Exp2_3_(30noise_45_10gripper)_2021-05-06-20-15-09.bag')
#exp = 23

#b = bagreader('bagfiles/Exp3_1_(10noise_30_10gripper)_2021-05-06-20-41-56.bag')
#exp = 31

#b = bagreader('bagfiles/Exp3_2_(20noise_30_10gripper)_2021-05-06-20-49-54.bag')
#exp = 32

#b = bagreader('bagfiles/Exp3_3_(30noise_30_10gripper)_2021-05-06-20-54-23.bag')
#exp = 33

#b = bagreader('bagfiles/Exp4_1_(0noise_45_0gripper)_2021-05-06-20-59-02.bag')
#exp = 41

#b = bagreader('bagfiles/Exp4_2_(0noise_45_5gripper)_2021-05-06-21-05-03.bag')
#exp = 42

#b = bagreader('bagfiles/Exp4_3_(0noise_45_10gripper)_2021-05-06-21-08-20.bag')
#exp = 43

#b = bagreader('bagfiles/Exp4_4_(0noise_45_15gripper)_2021-05-06-21-13-37.bag')
#exp = 44  
batch = '12'
b = bagreader('Deep_Learning_Data/deep_learning_data_'+batch+'.bag')
       
# Get the list of topics available in the file
print(b.topic_table)
    
# Read each topic
# a. Wrench topic: forces and torques
data_f = b.message_by_topic('wrench')
forces = pd.read_csv(data_f)
# b. Joint_States topic: angular positions of the 6 joints
data_j = b.message_by_topic('joint_states')
positions = pd.read_csv(data_j)

data_l = b.message_by_topic('deep_learning_label')
labels_msg = pd.read_csv(data_l)

data_q = b.message_by_topic('deep_learning_quaternion')
quat = pd.read_csv(data_q)

time_stamp = forces.iloc[:,0]
forces_x = forces.iloc[:,5]
forces_y = forces.iloc[:,6]
forces_z = forces.iloc[:,7]

torques_x = forces.iloc[:,8]
torques_y = forces.iloc[:,9]
torques_z = forces.iloc[:,10]

joint_0_pos = positions.iloc[:,6]

label_time = labels_msg.iloc[:,0]
labels = labels_msg.iloc[:,1]

quat_data = quat.iloc[:,2:]

print(labels)

sigma = 3 # this depends on how noisy your data is, play with it!

forces_x = gaussian_filter(forces_x, sigma)
forces_y = gaussian_filter(forces_y, sigma)
forces_z = gaussian_filter(forces_z, sigma)

net_force = [None] * len(forces)
elapsed_time = [None] * len(forces)
indexes = [None] * len(forces)
positions = [None] * len(forces)






# Compute the net values
for i in range(len(forces_z)):        
    net_force[i] = math.sqrt(forces_x[i]**2 + forces_y[i]**2 + forces_z[i]**2)   
    # Remove the time offset of the time stamp
    elapsed_time[i] = time_stamp[i] - time_stamp[0]    
    indexes[i] = i



# -----------------------------

success_max_forces =[]  
unsuccess_max_forces = []
unsuccess_mean = 0
unsuccess_std = 0
start_time = time_stamp[0]
time_stamp = time_stamp - start_time
print(time_stamp[0])
label_time = label_time - start_time


fig, ax = bagpy.create_fig(2)
last_ind = len(time_stamp)-1
count=0
t = [0.5,20]
a0 = 5
f0 = 3
delta_f = 5.0
s = [4,60]
ax[0].set_xlim([0,time_stamp[last_ind]])
ax[0].set_ylim([min(forces_z),max(forces_z)])
ax[1].set_xlim([0,time_stamp[last_ind]])
ax[1].set_ylim([min(forces_y),max(forces_y)])
X,Y,Z=[0,0,1],[0,0,1],[0,0.3,1]

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
axZ = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axlen = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

#ax.margins(x=0)

#and these set up the sliders

sfreq = Slider(axfreq, 'Prev Trial End', 0, time_stamp[last_ind], valinit=label_time[count])
samp = Slider(axamp, 'Grasp Start', 0, time_stamp[last_ind], valinit=label_time[count+1])
sZ = Slider(axZ,'End Pull',0,time_stamp[last_ind],valinit=label_time[count+2])

def update(val):
    release_pos = samp.val
    start_pos = sfreq.val
    end_pos = sZ.val
    ax[0].clear()
    ax[1].clear()
    ax[0].axvline(start_pos,color='r')
    ax[0].axvline(release_pos,color='b')
    ax[0].axvline(end_pos,color='g')
    ax[1].axvline(start_pos,color='r')
    ax[1].axvline(release_pos,color='b')
    ax[1].axvline(end_pos,color='g')
    ax[0].scatter(time_stamp,forces_z, label = 'Force z')
    ax[1].scatter(time_stamp,forces_y, label = 'Force y')
    ax[0].set_xlim([start_pos-15,end_pos+15])
    ax[1].set_xlim([start_pos-15,end_pos+15])
    fig.canvas.draw_idle()    

sfreq.on_changed(update)
samp.on_changed(update)
sZ.on_changed(update)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Next', color=axcolor, hovercolor='0.975')

def next_state(event):
    global count
    label_time[count] = sfreq.val
    label_time[count+1] = samp.val
    label_time[count+2] = sZ.val
    count+=3
    sfreq.reset()
    samp.reset()
    sZ.reset()
    print(count,len(labels))
    if count >= len(labels):
        print('finished this dataset, saving now')
        big_labels = time_stamp.copy()
        c = -1
        q_count = -1
        big_quats = np.ones((4,len(time_stamp)))
        label_time[30] = time_stamp[last_ind]
        for i,v in enumerate(time_stamp):
            if v > label_time[c+1]:
                print(i,v,label_time[c+1],c,q_count)
                c += 1
                if c % 3 == 0:
                    q_count += 1
            if (c > 0)&(c < 31):
                big_labels[i] = labels[c]
                big_quats[:,i] = [quat_data.data_0[q_count],quat_data.data_1[q_count],quat_data.data_2[q_count],quat_data.data_3[q_count]]
            else:
                big_labels[i] = -1
                big_quats[:,i] = [-1,-1,-1,-1]
        print('load next episode')
        filename = "/home/orochi/Downloads/Deep_Learning_Data/Batch" + batch
        print("Saving...")
        data = {}
        data["states"] = np.vstack((forces_x,forces_y,forces_z))
        data["labels"] = big_labels
        data["quats"] = big_quats
        print(data["states"][:,0:3])
        print(big_labels[0:3])
        print(big_quats[0:3])

        file = open(filename + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".pkl", 'wb')
        pkl.dump(data, file)
        file.close()
    sfreq.val = label_time[count]
    samp.val = label_time[count+1]
    sZ.val = label_time[count+2]
    
button.on_clicked(next_state)
button.on_clicked(update)
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)

plt.show()