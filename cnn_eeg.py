# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:57:38 2019

@author: LeoChu
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from load_data import load_train_data
#from load_data import load_test_data
from keras.utils import np_utils
import random


class cnn_for_eeg(object):
    def __init__(self):
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.learing_rate = tf.train.natural_exp_decay(
                                                  learning_rate=0.97, 
                                                  global_step= global_step, 
                                                  decay_steps=10, 
                                                  decay_rate=0.9, 
                                                  staircase=True)
        self.n_classes = 3
        self.data_len = 3600
        self.batch_size = 100
#        X_ = tf.placeholder(tf.float32, [None, self.data_len])
#        self.x_input = tf.reshape(X_, [-1, 60, 60, 1])
        self.x_input = tf.placeholder(tf.float32, [None, 60, 60, 1])
        self.label = tf.placeholder(tf.int32, shape=[None, self.n_classes]) 
        self.keep_prob = tf.constant(0.75)
        self.n_test = 1500
        self.training=False
        self.k_size = 3 # convolutional core size
        self.in_channels = 1
        self.stride = 2
    
    def shuffleing(self, dataSet, lbl):
        index = [ii for ii in range(len(dataSet))]
        random.shuffle(index)
        shuffled_dataSet = dataSet[index]
        shuffled_labels = lbl[index]
        return shuffled_dataSet, shuffled_labels
        
    
        
    def get_datas(self):
        datas, lbls = load_train_data()       
        hot_labels = np_utils.to_categorical(lbls, self.n_classes)
        
        k1 = datas.shape[0]
        k2 = int(0.7*k1)
        
        data_train, data_test = datas[:k2], datas[k2:]
        label_train, label_test = hot_labels[:k2], hot_labels[k2:]   
#        print(data_train.shape)
        d_train,l_train = self.shuffleing(data_train, label_train)
        d_test,l_test = self.shuffleing(data_test, label_test)
        
        return d_train,l_train, d_test,l_test 
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
     
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
     
    def conv2d(self, x_input, Wgts):  # Wgts means weight
        return tf.nn.conv2d(x_input, Wgts, strides=[1,1,1,1], padding='SAME')
     
    # pooling 层
    def max_pool_3x3(self, x_input):
        pool_output = tf.nn.max_pool(x_input, 
                                     ksize=[1,3,3,1], 
                                     strides=[1,3,3,1], 
                                     padding='SAME')
        return pool_output        
        
        
    def network_building(self):
        ## layer 1st
        W_conv1 = self.weight_variable([self.k_size,self.k_size,self.in_channels,16])
        b_conv1 = self.bias_variable([16])
        y_conv1 = tf.nn.relu(self.conv2d(self.x_input, W_conv1) + b_conv1)        
        y_pool1 = self.max_pool_3x3(y_conv1)
        
        ## layer 2nd
        W_conv2 = self.weight_variable([self.k_size,self.k_size,16,32])
        b_conv2 = self.bias_variable([32])
        y_conv2 = tf.nn.relu(self.conv2d(y_pool1, W_conv2) + b_conv2)
        y_pool2 = self.max_pool_3x3(y_conv2)
        
        ## layer 3rd
        W_conv3 = self.weight_variable([self.k_size,self.k_size,32,32])
        b_conv3 = self.bias_variable([32])
        y_conv3 = tf.nn.relu(self.conv2d(y_pool2, W_conv3) + b_conv3)
        y_pool3 = self.max_pool_3x3(y_conv3)
        
        ## layer 4th
        W_conv4 = self.weight_variable([self.k_size,self.k_size,32,64])
        b_conv4 = self.bias_variable([64])
        y_conv4 = tf.nn.relu(self.conv2d(y_pool3, W_conv4) + b_conv4)
        y_pool4 = self.max_pool_3x3(y_conv4)
        
        ## layer 5th
        W_conv5 = self.weight_variable([self.k_size,self.k_size,64,64])
        b_conv5 = self.bias_variable([64])
        y_conv5 = tf.nn.relu(self.conv2d(y_pool4, W_conv5) + b_conv5)
        y_pool5 = self.max_pool_3x3(y_conv5)
        
        ## fully_connected layers
        feature_dim = y_pool5.shape[1] * y_pool5.shape[2] * y_pool5.shape[3]
        y_pool5 = tf.reshape(y_pool5, [-1, feature_dim])
        fc1 = tf.layers.dense(y_pool5, 1024, activation=tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
        dropout = tf.layers.dropout(fc2, 
                                    self.keep_prob, 
                                    training=self.training, 
                                    name='dropout')
        self.logit = tf.layers.dense(dropout, self.n_classes, name='logits')

    def lost(self):
        err = tf.losses.softmax_cross_entropy(onehot_labels = self.label, 
                                                  logits = self.logit)
        self.loss = tf.reduce_mean(err, name='loss')
    
    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.learing_rate).minimize(self.loss)
        
    def eval_(self):

        preds = tf.nn.softmax(self.logit)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.divide(tf.reduce_sum(tf.cast(correct_preds, tf.float32)), self.batch_size)

            
    def build(self):
        '''
        Build the computation graph
        '''
#        self.get_data()
        self.network_building()
        self.lost()
        self.optimize()
        self.eval_()
#        sefl.summary()

        
    def training_and_test(self, d_train,l_train, d_test,l_test):
        loss = []
        acc = []   
        epoches = 1000
        initial = tf.global_variables_initializer()
        session = tf.Session()
        session.run(initial)
        
        d_train_len, d_test_len = len(d_train), len(d_test)
        batches_in_train = int(d_train_len/self.batch_size)
        batches_in_test = int(d_test_len/self.batch_size)
        for jj in range(epoches):
            print("Epoach %d -th" % jj)
            for i in range(batches_in_train):
                self.training = True
                X_batch = d_train[i*self.batch_size:(i+1)*self.batch_size]
                Y_batch = l_train[i*self.batch_size:(i+1)*self.batch_size]
                X_batch = X_batch.reshape(X_batch.shape[0], 60, 60, 1)  
                
    #            loss_ = session.run([self.opt],
    #                                      feed_dict={self.x_input: X_batch, 
    #                                                 self.label: Y_batch})
                loss_, train_accuracy = session.run([self.opt, self.accuracy],
                                                        feed_dict={self.x_input: X_batch, 
                                                                   self.label: Y_batch})
                loss.append(loss_)
                if i % 100 == 0:               
                    print("training acc %g" % train_accuracy)
        
        
        for i in range(batches_in_test):
            self.training = False
            x_batch = d_test[i*self.batch_size:(i+1)*self.batch_size]
            y_batch = l_test[i*self.batch_size:(i+1)*self.batch_size]                
            x_batch = x_batch.reshape(x_batch.shape[0], 60, 60, 1)  
#            loss_ = session.run(self.opt, feed_dict={self.x_input: x_batch, 
#                                                     self.label: y_batch})
            _, test_acc = session.run([self.opt, self.accuracy],
                                                    feed_dict={self.x_input: x_batch, 
                                                               self.label: y_batch})
            acc.append(test_acc)
            if i % 100 == 0: 
                print(" test_accuracy %g" % test_acc)
                

#if __name__ == '__main__':
#    
#    model = cnn_for_eeg()
#    d_train,l_train, d_test,l_test = model.get_datas()
#    print(l_train.shape)
#    model.build()
#    model.training_and_test(d_train,l_train, d_test,l_test)
               
    
    
    
    
    
    
    
    
    
    
    