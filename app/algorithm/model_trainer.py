#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

    # perform train/valid split 
    train_data, valid_data = train_test_split(data, test_size=model_cfg['valid_split'])
    # print(train_data.shape, valid_data.shape) #; sys.exit()    

    # preprocess data
    print("Pre-processing data...")
    train_X, train_y, valid_X, valid_y , preprocessor = preprocess_data(train_data, valid_data, data_schema)              
    # print("train/valid data shape: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)    
    
    # balance the targetclasses  
    train_X, train_y = get_resampled_data(train_X, train_y)
    valid_X, valid_y = get_resampled_data(valid_X, valid_y)
    # print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)
    
    # Create and train model   
    model, history = train_model(train_X, train_y, valid_X, valid_y, hyper_params)    
    
    return preprocessor, model, history


def train_model(train_X, train_y, valid_X, valid_y, hyper_params):    
    # get model hyper-parameters parameters     
    data_based_params = get_data_based_model_params(train_X, train_y, valid_X, valid_y)
    model_params = { **data_based_params, **hyper_params }
    # print(model_params) ; sys.exit()    
    
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
        
    return model, history


def preprocess_data(train_data, valid_data, data_schema):    
    
    preprocessor = preprocess.DataPreProcessor(
        max_vocab_size=model_cfg["MAX_VOCAB_SIZE"],
        max_len=model_cfg["MAX_TEXT_LEN"],
        padding_pre=True,
        truncating_pre=True, 
        pad_value=0.
    )
    
    document_field = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["documentField"]  
    target_field = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["targetField"]  
    
    # fit on train data
    preprocessor.fit(texts = train_data[document_field], labels = train_data[target_field])
    
    # transform words to indexes and pad sequences to be equal length
    train_X, train_y = preprocessor.transform(texts = train_data[document_field], labels = train_data[target_field])
    valid_X, valid_y = preprocessor.transform(texts = valid_data[document_field], labels = valid_data[target_field])

    return train_X, train_y, valid_X, valid_y, preprocessor



def get_resampled_data(X, y):    
    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # resamples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time). 
    max_resample = model_cfg["max_resample_of_minority_classes"]
    
    # class_count = list(y.sum(axis=0))
    class_counts = np.asarray(np.unique(y, return_counts=True)).T
    max_obs_count = max(class_counts[:, 1])
    
    resampled_X, resampled_y = [], []
    for i, count in list(class_counts):
        count = int(count)
        if count == 0: continue
        # find total num_samples to use for this class
        size = max_obs_count if max_obs_count / count < max_resample else count * max_resample
        size = int(size)
        # print(i, count, size)
        # if observed class is 50 samples, and we need 125 samples for this class, 
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples
        full_samples = size // count
        idx = y == i
        for _ in range(full_samples):
            resampled_X.append(X[idx, :])
            resampled_y.append(y[idx])
            
        # find the remaining samples to draw randomly
        remaining =  int(size - count * full_samples   )
        sampled_idx = np.random.randint(count, size=remaining)
        resampled_X.append(X[idx, :][sampled_idx, :])
        resampled_y.append(y[idx][sampled_idx])
        
    resampled_X = np.concatenate(resampled_X, axis=0)
    resampled_y = np.concatenate(resampled_y, axis=0)
    # print(resampled_X.shape, resampled_y.shape)
    # shuffle the arrays
    resampled_X, resampled_y = shuffle(resampled_X, resampled_y)
    
    return resampled_X, resampled_y