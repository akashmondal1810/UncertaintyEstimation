#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import sys
import json

from preprocessing import Information, Preprocess

"""
Helper script to run the proprocessing functions
"""

class runPreprocess():

    def __init__(self, data_input_path):
        self._input_path = data_input_path
        self.data = None
        self.ppc_parameters = None

        self._preprocessor  =Preprocess()
        self._getInfo = Information()


        print("Please Ensure to fill up the 'preprocessing.json' file first!")
        
        _preprocessing_file_PATH = 'preprocessing.json'
        with open(_preprocessing_file_PATH) as f:
            preprocessing_param = json.load(f)

        self.ppc_parameters = preprocessing_param

    def _read_csv_file(self):
        col_to_read = self.ppc_parameters["col_to_load"]
        if len(col_to_read):
            return pd.read_csv(self._input_path, usecols=col_to_read, index_col=0)
        else:
            return pd.read_csv(self._input_path)

    def get_dataset_info(self):
        self.data = self._read_csv_file()
        self._getInfo.info(self.data)

    def start_preprocessing(self):
        self._strategy()
        return self.data
        

    def _strategy(self):
        drop_strategy = self.ppc_parameters["drop_strategy"]
        print(drop_strategy)
        
        if drop_strategy:
            self.data = self._preprocessor.drop(self.data, drop_strategy)

        fill_strategy = self.ppc_parameters["fill_strategy"]
        
        if fill_strategy:
            self.data = self._preprocessor.fillna(self.data, fill_strategy)

        self.data = self._preprocessor._label_encoder(self.data)

        pfcol = self.ppc_parameters["col_to_dummied"]
        
        if pfcol:
            self.data=self._preprocessor._get_dummies(self.data, prefered_columns=pfcol)
        else:
            self.data=self._preprocessor._get_dummies(self.data, prefered_columns=None)
