# -*- coding: utf-8 -*-
"""
Created on 26/04/2020

Data Generation
@author: Michael Plumaris
"""
from __future__ import unicode_literals
import numpy as np
import pandas as pd
from scipy.spatial import distance


testing_data=True

if testing_data:
    step = 10
    angles = np.arange(0, 69.5+step, step)
else:
    step = 2.5
    angles = np.arange(0, 70+step, step)

dep_times = []
dep_states = []
arr_times = []
arr_states = []
for cone_angle in angles:
    for idx, l_point in enumerate(['1', '3']):
        if idx == 0:
            dep_times.append(np.load('output/l' + l_point + '_' + str(cone_angle) + '_time.npy'))
            dep_states.append(np.load('output/l' + l_point + '_'+ str(cone_angle) + '_state.npy'))
        else:
            arr_times.append(np.load('output/l' + l_point + '_' + str(cone_angle) + '_time.npy'))
            arr_states.append(np.load('output/l' + l_point + '_'+ str(cone_angle) + '_state.npy'))

columns = ['angle', 't_u', 't_s','j']
df = pd.DataFrame(columns=columns)
for id_angle, angle in enumerate(angles):
    dist=distance.cdist(dep_states[id_angle], arr_states[id_angle])
    np.meshgrid(dep_times[id_angle],arr_times[id_angle])
    mesh = np.array(np.meshgrid(dep_times[id_angle],arr_times[id_angle]))
    combinations = mesh.T.reshape(-1, 2)
    dist_arr=dist.reshape(-1,1)
    data_base=np.hstack([np.ones(dist_arr.shape)*angle,combinations,dist_arr])
    df_temp=pd.DataFrame(data_base, columns=columns)
    df=df.append(df_temp,ignore_index=True)
print(df.shape)
df=df.reset_index()
if testing_data:
    df.to_csv('./Data/df_test.csv', index=False, header=False, encoding='utf-8')
else:
    df.to_csv('./Data/df_train.csv', index=False, header=False, encoding='utf-8')

