# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:12:36 2019

@author: LeoChu
"""


from cnn_eeg import cnn_for_eeg
        
if __name__ == '__main__':
    
    model = cnn_for_eeg()
    d_train,l_train, d_test,l_test = model.get_datas()
    print(l_train.shape)
    model.build()
    model.training_and_test(d_train,l_train, d_test,l_test)
            
        
        
        
        
        
        
        
        
        