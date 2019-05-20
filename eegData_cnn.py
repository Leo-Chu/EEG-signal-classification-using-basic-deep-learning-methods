# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:24:20 2016

@author: LeoChu
"""

from __future__ import print_function
import numpy as np
#import sys   
#sys.setrecursionlimit(100000)
np.random.seed(1337)  # for reproducibility

#import h5py 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from data_preproccess import load_train_data
from data_preproccess import svc
from data_preproccess import voting
from data_preproccess import rf
from data_preproccess import load_test_data
from keras.utils import np_utils#, generic_utils
import pickle
#import theano
import matplotlib.pyplot as plt

#from keras.callbacks import EarlyStopping

batch_size = 100
nb_classes = 3
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 63, 100
# number of convolutional filters to use
nb_filters = 30
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets

data_train, label_train = load_train_data()
data_test = load_test_data()

nb_class = 3
label = np_utils.to_categorical(label_train, nb_class)
#print("label=",label)

#index = [i for i in range(len(data_train))]
#random.shuffle(index)
#data_train = data_train[index]
#label = label[index]
#del index

#
data_train = data_train.reshape(data_train.shape[0], 1, img_rows, img_cols)
data_test = data_test.reshape(data_test.shape[0], 1, img_rows, img_cols)

if K.image_dim_ordering() == 'th':
    data_train = data_train.reshape(data_train.shape[0], 1, img_rows, img_cols)
    data_test = data_test.reshape(data_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    data_train = data_train.reshape(data_train.shape[0], img_rows, img_cols, 1)
    data_test = data_test.reshape(data_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    
k1 = data_train.shape[0]
k2 = int(0.9*k1)
(X_train,X_val) = (data_train[:k2],data_train[k2:])
(Y_train,Y_val) = (label[:k2],label[k2:])
del data_train, label    

X_train = X_train.astype('float32')
data_test = data_test.astype('float32')
X_train /= X_train.max()
data_test /= data_test.max()
print('data_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(data_test.shape[0], 'test samples')
#
#
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


#early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_val, Y_val))

#score = model.evaluate(X_val, Y_val, verbose=0)
pickle.dump(model,open("./model.pkl","wb"))

origin_model = pickle.load(open("model.pkl", "rb"))
pred_testlabel = origin_model.predict_classes(data_test, batch_size = batch_size, verbose = 1)


### svm and rf
#dd = np.concatenate((X_train, data_test))
#get_feature = theano.function([model.layers[0].input], model.layers[8].output,\
#                              allow_input_downcast=False)   
#feaure = get_feature(dd)
#
#pred_testlabel2 = svc(dd[0:k2],label_train,dd[k2:])
#pred_testlabel3 = rf(dd[0:k2],label_train,dd[k2:])
#######

#####  voting
#num = int(pred_testlabel.shape[0]/img_cols)
#acc1 = np.zeros(100)
#acc2 = np.zeros(100)
#acc3 = np.zeros(100)
#
#for kk in range(100):
#    pt = pred_testlabel[kk*num:(kk+1)*num] 
#    acc1[kk] = len([1 for i in range(num) if pt[i] == 0])/num
#    acc2[kk] = len([1 for i in range(num) if pt[i] == 1])/num
#    acc3[kk] = len([1 for i in range(num) if pt[i] == 2])/num

#np.argmax

pre_acc = voting(pred_testlabel,tp_num = 10,org_class = 0)
print('predict accuracy:',pre_acc)

#plt.plot(acc1,'k-+') #0
#plt.plot(acc2,'k-.') #1
#plt.plot(acc3,'k:') #2
#plt.xlabel('persons')
#plt.ylabel('Detection Rate')
#plt.show()

#test results：
#cc = [acc1, acc2, acc3]
#acc = np.array(cc).T
#ind = np.argmax(acc, axis=1)
#plt.plot(ind,'r--')
#
#get_feature = theano.function([model.layers[0].input],model.layers[1].output,\
#                              allow_input_downcast=False)   
#feaure = get_feature(X_train)

#plt.plot(y, 'cx--', y+1, 'mo:', y+2, 'kp-.');  

