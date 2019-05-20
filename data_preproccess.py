# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:45:10 2017

@author: LeoChu
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337) 
import os
import scipy.io as sio
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 



def load_data(path):
#    print(path)
    mats = os.listdir(path)
    row, totalCol, col = 63, 300000, 500
    num = int(len(mats) * totalCol / col)
    data = np.empty((num, row, row), dtype="uint8")
    x = 0
    for i in range(len(mats)):
        each_data = sio.loadmat(path + mats[i])
        dataMat = each_data["data"]
	
        colInd = np.linspace(0, totalCol, round(totalCol / col), endpoint = False, dtype = "int")
        for j in colInd:
#            arr = np.asarray(dataMat[:row, j *col : (j + 1)*col], dtype = "uint8")
            arr = dataMat[:row, j : (j + col)]
#            data[x, :, :] = image_graying(arr)
            data[x, :, :] = np.cov(arr) #- np.std(arr, axis=1) 
            x = x + 1


    data /= np.max(data)

    return data

def load_train_data():
#    folders = ["./HC/","./FES/","./CHR/"] # data1
    folders = ["./HC_filtered/","./FES_filtered/","./CHR_filtered/"] # data
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
#    del index
    return dataSet, label

def load_test_data():
    
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

def svc(traindata,trainlabel,testdata):  
    print("Start training SVM...")  
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=300)  
    svcClf.fit(traindata,trainlabel)       
    pred_testlabel = svcClf.predict(testdata)
    return pred_testlabel
  
def rf(traindata,trainlabel,testdata):  #,testlabel
    print("Start training Random Forest...")  
    rfClf = RandomForestClassifier(n_estimators=300,criterion='gini')  
    rfClf.fit(traindata,trainlabel)  
    pred_testlabel = rfClf.predict(testdata)
    return pred_testlabel
    
def voting(pred_label,tp_num,org_class): # tp_num = test people num
    
    num = int(pred_label.shape[0]/tp_num)
    acc1 = np.zeros(num)
    acc2 = np.zeros(num)
    acc3 = np.zeros(num)
    pred1 = np.zeros(tp_num)
    
    for kk in range(tp_num):
        pt = pred_label[kk*num:(kk+1)*num] 
        acc1[kk] = len([1 for i in range(num) if pt[i] == 0])/num
        acc2[kk] = len([1 for i in range(num) if pt[i] == 1])/num
        acc3[kk] = len([1 for i in range(num) if pt[i] == 2])/num
    
    acc = np.array([acc1,acc2,acc3])
    for ii in range(tp_num):
        pred1[ii] = np.argmax(acc[:,ii])
        
    pred_acc = len([1 for i in range(tp_num) if pred1[ii] == org_class])/tp_num
    return pred_acc




   
if __name__ == "__main__":
    data, label = load_train_data()
    data_test = load_test_data()
    


