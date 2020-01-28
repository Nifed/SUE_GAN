#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Bidirectional, LSTM,  Activation, GRU
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras import backend as K

npzfile = np.load('./data/fma_arrays/dataset.npz')
x  = npzfile['arr_0'] # Spectrograms arrays
y  = npzfile['arr_1']

y = y-1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid =  train_test_split(X_train, y_train, test_size=0.2, random_state=42)

y_train = keras.utils.to_categorical(y_train, num_classes=8)
y_test = keras.utils.to_categorical(y_test, num_classes=8)
y_valid = keras.utils.to_categorical(y_valid, num_classes=8)

def build_model(model_input):
    
    layer = model_input
    
    conv_1 = Conv2D(filters = 16, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu', name='conv_1')(layer)
    
    pool_1 = MaxPooling2D((2,2) )(conv_1)

    conv_2 = Conv2D(filters = 32, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu', name='conv_2')(pool_1)
    pool_2 = MaxPooling2D((2,2) )(conv_2)

    conv_3 = Conv2D(filters = 64, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu', name='conv_3')(pool_2)
    pool_3 = MaxPooling2D((2,2) )(conv_3)
    
    
    conv_4 = Conv2D(filters = 64, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu', name='conv_4')(pool_3)
    pool_4 = MaxPooling2D((4,4))(conv_4)
    
    
    conv_5 = Conv2D(filters = 64, kernel_size = (3,1), strides=1,
                      padding= 'valid', activation='relu', name='conv_5')(pool_4)
    pool_5 = MaxPooling2D((4,4))(conv_5)

    flatten1 = Flatten()(pool_5)

    pool_lstm1 = MaxPooling2D((4,2), name = 'pool_lstm')(layer)
    

    squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(pool_lstm1)
    
    lstm = Bidirectional(GRU(64))(squeezed)  
    
    concat = concatenate([flatten1, lstm], axis=-1, name ='concat')
    
    output = Dense(8, activation = 'softmax', name='preds')(concat)
    
    model_output = output
    model = Model(model_input, model_output)
    

    model.compile(
            loss='categorical_crossentropy',
            optimizer="RMSprop",
            metrics=['accuracy']
        )
    
    print(model.summary())
    return model


def train_model(x_train, y_train, x_val, y_val):
    
    n_frequency = 128
    n_frames = 640
    x_train = np.expand_dims(x_train, axis = -1)
    x_val = np.expand_dims(x_val, axis = -1)
    
    
    input_shape = (n_frames, n_frequency, 1)
    model_input = Input(input_shape, name='input')
    
    model = build_model(model_input)
   

    history = model.fit(x_train, y_train, batch_size=64, epochs=100,
                        validation_data=(x_val, y_val), verbose=1)

    return model, history


model, history  = train_model(X_train, y_train, X_valid, y_valid)




