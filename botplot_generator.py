#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:23:34 2022

@author: orochi
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt







# X20

full_proxy_20 = [[0.990259615384616,	0.969587628865979,	0.768270120259019,	0.8],
[0.990383547008547,	0.960824742268041,	0.870490286771508,	0.842857142857143],
[0.98088141025641,	0.95,	0.927844588344126,	0.871428571428571],
[0.990128205128205,	0.955154639175258,	0.882516188714154,	0.857142857142857]]

grasp_proxy_20 = [[0.851603098290598,	0.801030927835052,	0.748381128584644,	0.771428571428571],
[0.837821047008547,	0.791752577319588,	0.746530989824237,	0.742857142857143],
[0.812351495726496,	0.777319587628866,	0.753469010175763,	0.785714285714286],
[0.824663461538461,	0.801546391752577,	0.754856614246068,	0.757142857142857]]

pick_proxy_20 = [[0.987000534188034,	0.971134020618557,	0.876965772432932,	0.885714285714286],
[0.988281517094017,	0.96340206185567,	0.917668825161887,	0.885714285714286],
[0.98296688034188,	0.968556701030928,	0.904717853839038,	0.857142857142857],
[0.982225961538462,	0.966494845360825,	0.922294172062905,	0.885714285714286]]


# X10
full_proxy_10 = [[0.954689393939394,	0.923478260869565,	0.936170212765957,	0.9],
[0.965922727272727,	0.929565217391304,	0.887604070305273,	0.857142857142857],
[0.961472727272727,	0.921739130434783,	0.875115633672525,	0.857142857142857],
[0.976371212121212,	0.938260869565217,	0.883441258094357,	0.842857142857143]]


grasp_proxy_10 = [[0.829325757575757,	0.789565217391304,	0.735892691951896,	0.757142857142857],
[0.801174242424242,	0.777391304347826,	0.692876965772433,	0.742857142857143],
[0.82020303030303,	0.769565217391304,	0.684551341350601,	0.728571428571429],
[0.787624242424242,	0.74695652173913,	0.772432932469935,	0.771428571428571]]


pick_proxy_10 = [[0.973440909090909,	0.931304347826087,	0.839037927844588,	0.828571428571429],
[0.962778787878788,	0.935652173913044,	0.893154486586494,	0.857142857142857],
[0.958977272727273,	0.918260869565217,	0.886216466234968,	0.871428571428571],
[0.966518181818182,	0.923478260869565,	0.880666049953746,	0.871428571428571]]



# X5
full_proxy_5 = [[0.969014084507042,	0.937878787878788,	0.849213691026827,	0.828571428571429],
[0.95649041791734,	0.940909090909091,	0.877890841813136,	0.842857142857143],
[0.987056107134611,	0.945454545454545,	0.8159111933395,	0.828571428571429],
[0.976735165088894,	0.968181818181818,	0.802960222016651,	0.828571428571429]]

grasp_proxy_5 = [[0.789997691064419,	0.756060606060606,	0.726179463459759,	0.728571428571429],
[0.805855460632648,	0.783333333333333,	0.736355226641998,	0.771428571428571],
[0.789498960978989,	0.760606060606061,	0.729879740980573,	0.771428571428571],
[0.795617640267836,	0.760606060606061,	0.714153561517114,	0.728571428571429]]

pick_proxy_5 = [[0.986806742091896,	0.957575757575757,	0.789084181313598,	0.814285714285714],
[0.978185176633572,	0.942424242424242,	0.824236817761332,	0.771428571428571],
[0.971757099976911,	0.954545454545455,	0.835337650323774,	0.871428571428571],
[0.967033017778804,	0.943939393939394,	0.820536540240518,	0.842857142857143]]


# X1

full_proxy_1 = [[0.957647058823529,	0.898305084745763,	0.842738205365402,	0.828571428571429],
[0.939705882352941,	0.906779661016949,	0.590194264569843,	0.7],
[0.922352941176471,	0.906779661016949,	0.911193339500463,	0.885714285714286],
[0.925588235294118,	0.864406779661017,	0.896392229417206,	0.857142857142857]]

grasp_proxy_1 =[[0.766764705882353,	0.76271186440678,	0.719703977798335,	0.757142857142857],
[0.816176470588235,	0.779661016949153,	0.634597594819611,	0.7],
[0.800735294117647,	0.76271186440678,	0.686401480111008,	0.742857142857143],
[0.740882352941176,	0.728813559322034,	0.696577243293247,	0.728571428571429]]

pick_proxy_1 = [[0.900441176470588,	0.872881355932203,	0.775208140610546,	0.8],
[0.893088235294118,	0.864406779661017,	0.85013876040703,	0.828571428571429],
[0.952647058823529,	0.932203389830508,	0.880666049953746,	0.842857142857143],
[0.953823529411765,	0.923728813559322,	0.595744680851064,	0.757142857142857]]


# 25/75 train test split
#full_real = [[0.986111111111111,	0.944444444444444],
#[0.986111111111111,	0.944444444444444],
#[0.958333333333333,	0.944444444444444],
#[0.963636363636364,	0.9375]]
#
#grasp_real = [[0.930555555555556,	0.888888888888889],
#[0.833333333333333,	0.888888888888889],
#[0.847222222222222,	0.833333333333333],
#[0.927272727272727,	0.9375]]
#	
#pick_real = [[1,	1],
#[0.888888888888889,	0.888888888888889],
#[0.833333333333333,	0.944444444444444],
#[0.927272727272727,	0.9375]]


# 50/50 train test split
full_real = [[0.840277777777778,	0.888888888888889],
[0.869565217391304,	0.882352941176471],[0.934027777777778,	0.888888888888889],
[0.893280632411067,	0.852941176470588]]

grasp_real = [[0.729166666666667,	0.805555555555556],
[0.774703557312253,	0.764705882352941],[0.743055555555556,	0.777777777777778],
[0.675889328063241,	0.764705882352941]]
	
pick_real = [[0.850694444444444,	0.861111111111111],
[0.936758893280632,	0.911764705882353],[0.482638888888889,	0.75],
[0.83399209486166,	0.823529411764706]]


full_proxy_1 = np.array(full_proxy_1)
grasp_proxy_1 = np.array(grasp_proxy_1)
pick_proxy_1 = np.array(pick_proxy_1)
full_proxy_5 = np.array(full_proxy_5)
grasp_proxy_5 = np.array(grasp_proxy_5)
pick_proxy_5 = np.array(pick_proxy_5)
full_proxy_10 = np.array(full_proxy_10)
grasp_proxy_10 = np.array(grasp_proxy_10)
pick_proxy_10 = np.array(pick_proxy_10)
full_proxy_20 = np.array(full_proxy_20)
grasp_proxy_20 = np.array(grasp_proxy_20)
pick_proxy_20 = np.array(pick_proxy_20)
#full_proxy = np.array(full_proxy)
#grasp_proxy = np.array(grasp_proxy)
#pick_proxy = np.array(pick_proxy)
full_real = np.array(full_real)
grasp_real = np.array(grasp_real)
pick_real = np.array(pick_real)

#plot_arr = np.zeros([4,9])
#plot_arr[:,0] = grasp_proxy[:,1]
#plot_arr[:,1] = grasp_proxy[:,3]
#plot_arr[:,2] = grasp_real[:,1]
#plot_arr[:,3] = pick_proxy[:,1]
#plot_arr[:,4] = pick_proxy[:,3]
#plot_arr[:,5] = pick_real[:,1]
#plot_arr[:,6] = full_proxy[:,1]
#plot_arr[:,7] = full_proxy[:,3]
##plot_arr[:,8] = full_real[:,1]
#grasp_proxy_1 = np.array([0.7,0.7,0.7,0.7])
#pick_proxy_1 = np.array([0.7,0.7,0.7,0.7])
#full_proxy_1 = np.array([0.7,0.7,0.7,0.7])
#
#plot_auc_arr = np.zeros([4,9])
#plot_auc_arr[:,0] = grasp_proxy[:,0]
#plot_auc_arr[:,1] = grasp_proxy[:,2]
#plot_auc_arr[:,2] = grasp_real[:,0]
#plot_auc_arr[:,3] = pick_proxy[:,0]
#plot_auc_arr[:,4] = pick_proxy[:,2]
#plot_auc_arr[:,5] = pick_real[:,0]
#plot_auc_arr[:,6] = full_proxy[:,0]
#plot_auc_arr[:,7] = full_proxy[:,2]
#plot_auc_arr[:,8] = full_real[:,0]
#colors = [[0.00392156862745098, 0.45098039215686275, 0.6980392156862745],
# [0.8705882352941177, 0.5607843137254902, 0.0196078431372549],
# [0.00784313725490196, 0.6196078431372549, 0.45098039215686275],
# [0.00392156862745098, 0.45098039215686275, 0.6980392156862745],
# [0.8705882352941177, 0.5607843137254902, 0.0196078431372549],
# [0.00784313725490196, 0.6196078431372549, 0.45098039215686275],
# [0.00392156862745098, 0.45098039215686275, 0.6980392156862745],
# [0.8705882352941177, 0.5607843137254902, 0.0196078431372549],
# [0.00784313725490196, 0.6196078431372549, 0.45098039215686275]]

color_dict = {'x1':[0.00392156862745098, 0.45098039215686275, 0.6980392156862745],
 'x5':[0.8705882352941177, 0.5607843137254902, 0.0196078431372549],
 'x10':[0.00784313725490196, 0.6196078431372549, 0.45098039215686275],
 'x20':[0.8352941176470589, 0.3686274509803922, 0.0],
 'real':[0.8, 0.47058823529411764, 0.7372549019607844]}

plot_auc_arr = np.zeros([4,15])
plot_auc_arr[:,0] = grasp_proxy_1[:,2]
plot_auc_arr[:,1] = grasp_proxy_5[:,2]
plot_auc_arr[:,2] = grasp_proxy_10[:,2]
plot_auc_arr[:,3] = grasp_proxy_20[:,2]
plot_auc_arr[:,4] = grasp_real[:,0]
plot_auc_arr[:,5] = pick_proxy_1[:,2]
plot_auc_arr[:,6] = pick_proxy_5[:,2]
plot_auc_arr[:,7] = pick_proxy_10[:,2]
plot_auc_arr[:,8] = pick_proxy_20[:,2]
plot_auc_arr[:,9] = pick_real[:,0]
plot_auc_arr[:,10] = full_proxy_1[:,2]
plot_auc_arr[:,11] = full_proxy_5[:,2]
plot_auc_arr[:,12] = full_proxy_10[:,2]
plot_auc_arr[:,13] = full_proxy_20[:,2]
plot_auc_arr[:,14] = full_real[:,0]
colors = [color_dict['x1'],color_dict['x5'],color_dict['x10'],color_dict['x20'],color_dict['real'],
          color_dict['x1'],color_dict['x5'],color_dict['x10'],color_dict['x20'],color_dict['real'],
          color_dict['x1'],color_dict['x5'],color_dict['x10'],color_dict['x20'],color_dict['real']]
#sns.boxplot(data=plot_arr, palette = colors)

sns.boxplot(data=plot_auc_arr, palette = colors)

#plt.boxplot(full_proxy[:,0:2])
#plt.boxplot(grasp_proxy[:,0:2])
#plt.boxplot(pick_proxy[:,0:2])