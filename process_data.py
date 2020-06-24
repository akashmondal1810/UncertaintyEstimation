#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import sys
import json
import tensorflow as tf

from preprocessing.pp_runner import runPreprocess


datasetPath = sys.argv[1]
datasetOutputPath = sys.argv[2]


ppr = runPreprocess(datasetPath)
ppr.get_dataset_info()
datss = ppr.start_preprocessing()
    
train, validate, test = np.split(datss.sample(frac=1), [int(.6*len(datss)), int(.8*len(datss))])
    
train_path = datasetOutputPath+'/train.csv'
train.to_csv(train_path, index=False, encoding='utf-8')
print('Train data saved at: ', train_path)
    
validate_path = datasetOutputPath+'/validate.csv'
validate.to_csv(validate_path, index=False, encoding='utf-8')
print('validate data saved at: ', validate_path)

test_path = datasetOutputPath+'/test.csv'
test.to_csv(test_path, index=False, encoding='utf-8')
print('test data saved at: ', test_path)