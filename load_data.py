# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:40:13 2019

@author: LeoChu
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337) 
import os
import scipy.io as sio
import random


def load_data(path):
#    print(path)
    mats = os.listdir(path)
    row, totalCol = 60, 300000
    samples = round(totalCol / row)
    num = int(len(mats) * totalCol / row)
    data = np.empty((num, row, row), dtype="float32")
    x = 0
    for i in range(len(mats)):
        each_data = sio.loadmat(path + mats[i])
        dataMat0 = each_data["data1"]
        dataMat = dataMat0[:row, :totalCol]
#        print(dataMat.shape())
#        colInd = np.linspace(0, totalCol, samples, endpoint = False, dtype = "int")
        for j in range(samples):
            arr = np.asarray(dataMat[:, j *row : (j + 1)*row], dtype = "uint8")
            data[x, :, :] = arr
            x = x + 1
            
#    data /= np.max(data)

    return data

def load_train_data():
#    folders = ["./HC/","./FES/","./CHR/"] # data1
    folders = ["./CHR/","./FES/","./HC/"] # data
    folders_num = len(folders)
    dataSet = np.empty((), dtype="uint8")
    label = np.empty((), dtype="uint8")

    for i in range(folders_num):
        data = load_data(folders[i])
        each_label = np.empty((data.shape[0],1), dtype="uint8")
        each_label.fill(i)
        if i == 0:
            dataSet, label = data, each_label
        else:
#            print(dataSet.shape)
            dataSet = np.concatenate((dataSet, data))
            label = np.concatenate((label, each_label))
#    index = [ii for ii in range(len(dataSet))]
#    random.shuffle(index)
#    dataSet = dataSet[index]
#    label = label[index]
    return dataSet, label

def load_test_data0():
    
   folders = ["./datatest-hc/"]
   folders_num = len(folders)
   dataSet = np.empty((), dtype="uint8")

   for i in range(folders_num):
       data = load_data(folders[i])
       if i == 0:
           dataSet = data 
       else:
           dataSet = np.concatenate((dataSet, data))
   return dataSet

   
   
def load_test_data():   
    folders = ["./datatest-hc/"] # data
    folders_num = len(folders)
    dataSet = np.empty((), dtype="uint8")
    label = np.empty((), dtype="uint8")

    for i in range(folders_num):
        data = load_data(folders[i])
        each_label = np.empty((data.shape[0],1), dtype="uint8")
        each_label.fill(i)
        if i == 0:
            dataSet, label = data, each_label
        else:
            print(dataSet.shape)
            dataSet = np.concatenate((dataSet, data))
            label = np.concatenate((label, each_label))
    index = [ii for ii in range(len(dataSet))]
    random.shuffle(index)
    dataSet = dataSet[index]
    label = label[index]
    return dataSet, label