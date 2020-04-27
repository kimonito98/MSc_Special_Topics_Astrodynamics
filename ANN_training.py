# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:47:38 2020

@author: Michael
"""
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


 

def build_model(input_len):

    model = keras.Sequential([
         layers.Dense(128, activation='tanh', input_shape=[input_len]),
         layers.Dense(64, activation='relu'),
         layers.Dense(64, activation='relu'),
         layers.Dense(32, activation='relu'),
         layers.Dense(32, activation='relu'),
         layers.Dense(1)
    ])
    


    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    model.compile(loss='mape',
                optimizer='adam',
                metrics=['mape', 'mae','mse', 'RootMeanSquaredError'])
    return model

def normalise(df):
    raw_dataset['angle']=raw_dataset['angle']/70.
    raw_dataset['t_u']=raw_dataset['t_u']/51.06789
    raw_dataset['t_s']=raw_dataset['t_s']/51.06789
    return df



#%%Load train  (every 2.5 deg)  data from csv files , drop index column and normalise

column_names = ['Index','angle','t_u','t_s','j']

raw_dataset=pd.read_csv(r'C:\Users\Michael\Desktop\Special Topics\Data/df_train.csv', names=column_names,
                      sep=",", skipinitialspace=True).drop('Index',axis=1)

dataset=normalise(raw_dataset.copy())
train_dataset = dataset.sample(frac=0.99,random_state=0) 
train_labels = train_dataset.pop('j')
dataset.describe()


#%%Load test (every 10 deg) data from csv files , drop index column and normalise

test_dataset=pd.read_csv(r'C:\Users\Michael\Desktop\Special Topics\Data/df_test.csv', names=column_names,
                      sep=",", skipinitialspace=True).drop('Index',axis=1)
test_dataset=normalise(test_dataset)
test_labels = test_dataset.pop('j')


#%% Build Model

model = build_model(len(train_dataset.keys()))



#%% Save weights
checkpoint_path = r"C:\Users\Michael\Desktop\Special Topics\check/cp.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)


#%% Train Model

epochs = 40
batch_size = 2048*2


history = model.fit(
      train_dataset, train_labels,
      epochs=epochs, validation_split = 0.2,verbose=1,batch_size=batch_size,callbacks=[cp_callback])

model.save_weights(checkpoint_path)



#%% Check Loss, RMS
plt.plot(history.history['loss'], 'k')
plt.plot(history.history['val_loss'], 'r')
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss','Validation Loss'],loc='best')


#%% Plot Prediction. Trend: linear--> good

model.load_weights(checkpoint_path)
res=model.predict(test_dataset.sample(frac=0.01,random_state=0)).flatten() 
plt.figure()
plt.scatter(test_labels.sample(frac=0.01,random_state=0),res,s=0.08,label='Predicted', color='cyan')
plt.xlabel('Euclidean Norm  [-]')
plt.ylabel('Estimated Euclidean Norm  [-]')
plt.xlim([min(test_labels),max(test_labels)])
plt.ylim([min(test_labels),max(test_labels)])
plt.plot([min(test_labels),max(test_labels)],[min(test_labels),max(test_labels)],color='red',label='Correct')
plt.grid(True)
plt.legend()


#%% Compare agains valid

test_dataset1=pd.read_csv(r'C:\Users\Michael\Desktop\Special Topics\Data/df_test.csv', names=column_names,
                      sep=",", skipinitialspace=True).drop('Index',axis=1)

test_dataset1=normalise(test_dataset1)
test_dataset1=test_dataset1.sample(frac=0.001,random_state=0)
test_labels1 = test_dataset1.pop('j')
model = build_model(len(test_dataset1.keys()))

model.load_weights(checkpoint_path)
result = model.predict(test_dataset1).flatten()
real_res= test_labels1
test_dataset1['Error']=result-real_res



#%% make boxplots

bplot = sns.boxplot(y='Error', x='angle', 
                 data=test_dataset1, 
                 width=0.5,
                 palette="colorblind")


