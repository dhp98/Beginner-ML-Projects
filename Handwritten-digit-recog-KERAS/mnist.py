#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 23:32:31 2018

@author: dhyey
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.datasets import mnist
from keras.utils import np_utils
#import  and split dataset
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = np.expand_dims(x_train,axis=0)
from keras.preprocessing.image import ImageDataGenerator
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test,10)
#initialize my CNN
classifier = Sequential()

#convolution and max pooling layers
classifier.add(Convolution2D(32, 3, activation = 'relu' ,input_shape=(28,28,1)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32 ,3,activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

#2 fully connected layers
classifier.add(Dense(128 , activation ='relu'))
classifier.add(Dropout(0.8))
#output layer
classifier.add(Dense(10 , activation="softmax"))

#compiling CNN model
classifier.compile(optimizer="adam" , loss="categorical_crossentropy" ,metrics=['accuracy'])

#nb_epoch = 10
#num_classes = 10
#batch_size = 128
#train_size = 60000
##test_size = 10000
#v_length = 784


#image - preprocessing and model fitting
#x_train = x_train[:10000 , :]
#x_test = x_test[:2000 , :]
#y_train = y_train[:10000]
#y_test = y_test[:2000 ]

x_train = np.expand_dims(x_train,axis=3)
x_test = np.expand_dims(x_test,axis=3)

classifier.fit(x_train, 
				 	y_train,
					validation_data=(x_test, y_test),
					batch_size=128,
					nb_epoch=10,
					verbose=2)












