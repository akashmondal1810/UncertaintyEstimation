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


datasetName = sys.argv[1]
datasetPath = sys.argv[2]

if datasetName=='loan':
    ppr = runPreprocess(datasetPath)
    ppr.get_dataset_info()
    datss = ppr.start_preprocessing()
    
    train, validate, test = np.split(datss.sample(frac=1), [int(.6*len(datss)), int(.8*len(datss))])
    
    train.to_csv('data/loan/train.csv', index=False, encoding='utf-8')
    print('Train data saved at data/loan/train.csv')
    validate.to_csv('data/loan/validate.csv', index=False, encoding='utf-8')
    print('validate data saved at data/loan/validate.csv')
    test.to_csv('data/loan/test.csv', index=False, encoding='utf-8')
    print('test data saved at data/loan/test.csv')