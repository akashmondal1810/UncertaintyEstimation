#!/usr/bin/env python
# coding: utf-8


"""
This script deals with the preprocessing of data and getting it 
ready for the machine learning algorithms. The main topics are:
1. Dealing with missing data
2. Handling categorical data
3. Bringing features on the same scale.
4. Selecting meaningful features.
5. Regularizing data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import sys
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler




class Information():

    def __init__(self):
        """
        This class give some brief information about the datasets.
        Information introduced in R language style
        """
        print("Information object created")

    def _get_missing_values(self,data):
        """
        Find missing values of given data
        :param data: checked its missing value
        :return: Pandas Series object
        """
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()

        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

    def _get_basic_info(silf, data):
        """
        Find shape and memory used of given data
        :param data: checked its missing value
        :return: list object
        """

        memory_size = sys.getsizeof(data)
        rows_count, cols_count = data.shape
        return [rows_count, cols_count, memory_size]

    def info(self,data):
        """
        print basic info, feature name, data type, number of missing values and ten samples of 
        each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """

        nRows, nCols, memoryUsed = self._get_basic_info(data)
        print('No of rows: ', nRows)
        print('No of columns: ', nCols)
        print('Memory Used: ', memoryUsed)


        feature_dtypes=data.dtypes
        self.missing_values=self._get_missing_values(data)

        print("=" * 60)

        print("{:16} {:16} {:25} {:16}".format("Feature Name".upper(),
                                            "Data Format".upper(),
                                            "# of Missing Values".upper(),
                                            "10 Samples".upper()))
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print("{:18} {:19} {:19} ".format(feature_name, str(dtype), str(missing_value)), end="")
            for v in data[feature_name].values[:10]:
                print(v, end=",")
            print()

        print("="*60)




class Preprocess():

    def __init__(self):
        print("Preprocess object created")

    def fillna(self, data, fill_strategies):
        """
        Fill missing values for each column using 'Zero', or feature 'Mode', 'Mean', or 'Median'
        :param data: dataset information will be gathered from
        :fill_strategies: strategies of filling missing value e.g. 'Mode', 'Mean', or 'Median' etc.
        :return: the processed dataframe
        """
        for column, strategy in fill_strategies.items():
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'Mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'Median':
                data[column] = data[column].fillna(data[column].median())
            else:
                print("{}: There is no such thing as preprocess strategy".format(strategy))

        return data

    def drop(self, data, drop_strategies):
        """
        Dropping the non informative features
        :param data: dataset information will be gathered from
        :fill_strategies: strategies of filling missing value e.g. 'Mode', 'Mean', or 'Median' etc.
        :return: the processed dataframe
        """
        for column in drop_strategies:
            print(column)
            if drop_strategies[column]==1:
                data=data.drop([column], axis=1)
            if drop_strategies[column]==0:
                data = data[data[column].notna()]

        return data


    def _label_encoder(self,data):
        """
        Encoding the categorical variable
        :param data: dataset information will be gathered from
        :return: the processed dataframe
        """

        cols = data.columns
        num_cols = data._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        

        labelEncoder=LabelEncoder()
        for column in cat_cols:
            print(column)
            labelEncoder.fit(data[column])
            data[column]=labelEncoder.transform(data[column])
        
        return data

    def _get_dummies(self, data, prefered_columns=None):
        """
        To get dummy variable
        :param data: dataset information will be gathered from
        :prefered_columns: features to be dummied, if set to None all feature will be dummied
        :return: the processed dataframe
        """

        if prefered_columns is None:
            columns=data.columns.values
            non_dummies=None
        else:
            non_dummies=[col for col in data.columns.values if col not in prefered_columns]

            columns=prefered_columns


        dummies_data=[pd.get_dummies(data[col],prefix=col) for col in columns]

        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append(data[non_dummy])

        return pd.concat(dummies_data, axis=1)

    def _get_normalized(self, data, norm_col):

        """
        For data normalization
        :param data: dataset information will be gathered from
        :return: the processed dataframe
        """
        data[norm_col] = StandardScaler().fit_transform(data[norm_col])
        return data
