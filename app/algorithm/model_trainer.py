#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

import algorithm.preprocessing.preprocess as preprocess
import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.utils as utils
from algorithm.model.classifier import Classifier, get_data_based_model_params
from algorithm.utils import get_model_config




# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # balance the target classes  
    data = utils.get_resampled_data(data = data, 
                        max_resample = model_cfg["max_resample_of_minority_classes"])
    # print(data.head()); sys.exit()        
    
    # perform train/valid split 
    train_data, valid_data = train_test_split(data, test_size=model_cfg['valid_split'])
    # print(train_data.shape, valid_data.shape) #; sys.exit()    

    # preprocess data
    print("Pre-processing data...")
    train_X, train_y, valid_X, valid_y , preprocessor = preprocess_data(train_data, valid_data)              
    # print("train/valid data shape: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)
    
    # Create and train model   
    model, history = train_model(train_X, train_y, valid_X, valid_y, hyper_params)    
    
    return preprocessor, model, history


def train_model(train_X, train_y, valid_X, valid_y, hyper_params):    
    # get model hyper-parameters parameters 
    
    data_based_params = get_data_based_model_params(train_X, train_y, valid_X, valid_y)
    model_params = { **data_based_params, **hyper_params }
    # print(model_params) #; sys.exit()
    
    # Create and train model   
    model = Classifier(  **model_params )  
    # model.summary()  ; sys.exit()
    
      
    print('Fitting model ...')  
    history = model.fit(
        train_X=train_X, train_y=train_y, 
        valid_X=valid_X, valid_y=valid_y,
        batch_size = 32, 
        epochs = 1000,
        verbose = 1, 
    )      
    # print("last_loss:", history.history['loss'][-1])
    return model, history


def preprocess_data(train_data, valid_data):    
    
    preprocessor = preprocess.DataPreProcessor(
        max_vocab_size=model_cfg["MAX_VOCAB_SIZE"],
        max_len=model_cfg["MAX_TEXT_LEN"],
        padding_pre=True,
        truncating_pre=True, 
        pad_value=0.
    )
    
    # fit on train data
    preprocessor.fit(texts = train_data['text'], labels = train_data['class'])
    
    # transform words to indexes and pad sequences to be equal length
    train_X, train_y = preprocessor.transform(texts = train_data['text'], labels = train_data['class'])
    valid_X, valid_y = preprocessor.transform(texts = valid_data['text'], labels = valid_data['class'])

    return train_X, train_y, valid_X, valid_y, preprocessor


