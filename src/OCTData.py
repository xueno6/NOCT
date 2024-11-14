# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:12:06 2024

@author: MSI-NB
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from BasisFunctions import BasisFunction
from Generators import ValueGenerator, ExpressionGenerator
class OCTData:
    @abstractmethod
    def __init__(self, df, train_df, validation_df, test_df):
        self.df = df
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df

    def insert(self, **kwargs):
        pass
    
    def mix(self, seed):
        concatenated_df = pd.concat([self.train_df, self.validation_df])

        # Set the random seed
        np.random.seed(seed)
        
        # Shuffle the rows
        shuffled_df = concatenated_df.sample(frac=1, random_state=seed)

        # Partition the shuffled DataFrame
        self.train_df = shuffled_df.iloc[:len(self.train_df)]
        self.validation_df = shuffled_df.iloc[len(self.train_df):]

    
class OCTData_quadratic(OCTData):
    def __init__(self,  df, train_df, validation_df, test_df):
        super().__init__(df, train_df, validation_df, test_df)
        
    def insert(self, feature_pairs):
        for fp in feature_pairs:
            fpDict = {}
            lst = list(fp)
            if len(lst)>1:
                fpDict['PLY-1, 1']= lst
            fpDict['PLY-2'] = lst
            VG=ValueGenerator(fpDict)
            VEG=ExpressionGenerator(fpDict)
            train_df_temp = pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            validation_df_temp = pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            test_df_temp =pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            
            for i in self.train_df.index:
                train_df_temp.loc[i] = VG(list(self.train_df.loc[i][:-1]))
            
            for i in self.validation_df.index:
                validation_df_temp.loc[i] = VG(list(self.validation_df.loc[i][:-1]))

            for i in self.test_df.index:
                test_df_temp.loc[i] = VG(list(self.test_df.loc[i][:-1]))
            
            insert_position = self.train_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.train_df.iloc[:, :insert_position]
            df1_last = self.train_df.iloc[:, insert_position:]
            self.train_df = pd.concat([df1_left, train_df_temp, df1_last], axis=1)
            
            insert_position = self.validation_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.validation_df.iloc[:, :insert_position]
            df1_last = self.validation_df.iloc[:, insert_position:]
            self.validation_df = pd.concat([df1_left, validation_df_temp, df1_last], axis=1)
            
            insert_position = self.test_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.test_df.iloc[:, :insert_position]
            df1_last = self.test_df.iloc[:, insert_position:]
            self.test_df = pd.concat([df1_left, test_df_temp, df1_last], axis=1)

class OCTData_aggressive(OCTData):
    def __init__(self,  df, train_df, validation_df, test_df):
        super().__init__(df, train_df, validation_df, test_df)
        
    def insert(self, feature_pairs):
        for fp in feature_pairs:
            fpDict = {}
            lst = list(fp)
            if len(lst)>1:
                fpDict['PLY-1, 1']= lst
                fpDict['PLZ-1, 2'] = lst
                #fpDict['USE-(X1+1)/(X2+1)'] = lst
            fpDict['PLY-2'] = lst
            fpDict['PLY-3'] = lst
            fpDict['SIN-0.3183098861837907'] = lst
            fpDict['COS-0.3183098861837907'] = lst
            VG=ValueGenerator(fpDict)
            VEG=ExpressionGenerator(fpDict)
            train_df_temp = pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            validation_df_temp = pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            test_df_temp =pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            
            for i in self.train_df.index:
                train_df_temp.loc[i] = VG(list(self.train_df.loc[i][:-1]))
            
            for i in self.validation_df.index:
                validation_df_temp.loc[i] = VG(list(self.validation_df.loc[i][:-1]))

            for i in self.test_df.index:
                test_df_temp.loc[i] = VG(list(self.test_df.loc[i][:-1]))
            
            insert_position = self.train_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.train_df.iloc[:, :insert_position]
            df1_last = self.train_df.iloc[:, insert_position:]
            self.train_df = pd.concat([df1_left, train_df_temp, df1_last], axis=1)
            
            insert_position = self.validation_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.validation_df.iloc[:, :insert_position]
            df1_last = self.validation_df.iloc[:, insert_position:]
            self.validation_df = pd.concat([df1_left, validation_df_temp, df1_last], axis=1)
            
            insert_position = self.test_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.test_df.iloc[:, :insert_position]
            df1_last = self.test_df.iloc[:, insert_position:]
            self.test_df = pd.concat([df1_left, test_df_temp, df1_last], axis=1)
            
            self.train_df =  self.train_df.loc[:,~ self.train_df.columns.duplicated()].copy()
            self.validation_df = self.validation_df.loc[:,~ self.validation_df.columns.duplicated()].copy()
            self.test_df = self.test_df.loc[:,~ self.test_df.columns.duplicated()].copy()
    def insert2(self, feature_pairs):
        for fp in feature_pairs:
            fpDict = {}
            lst = list(fp)
            if len(lst)>1:
                fpDict['PLY-1, 1']= lst
                #fpDict['USE-(X1+1)/(X2+1)'] = lst
            fpDict['PLY-2'] = lst
            fpDict['USE-np.sqrt(np.abs(X1))'] = lst    
            VG=ValueGenerator(fpDict)
            VEG=ExpressionGenerator(fpDict)
            train_df_temp = pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            validation_df_temp = pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            test_df_temp =pd.DataFrame(columns=VEG(list(self.train_df.columns)))
            
            for i in self.train_df.index:
                train_df_temp.loc[i] = VG(list(self.train_df.loc[i][:-1]))
            
            for i in self.validation_df.index:
                validation_df_temp.loc[i] = VG(list(self.validation_df.loc[i][:-1]))

            for i in self.test_df.index:
                test_df_temp.loc[i] = VG(list(self.test_df.loc[i][:-1]))
            
            insert_position = self.train_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.train_df.iloc[:, :insert_position]
            df1_last = self.train_df.iloc[:, insert_position:]
            self.train_df = pd.concat([df1_left, train_df_temp, df1_last], axis=1)
            
            insert_position = self.validation_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.validation_df.iloc[:, :insert_position]
            df1_last = self.validation_df.iloc[:, insert_position:]
            self.validation_df = pd.concat([df1_left, validation_df_temp, df1_last], axis=1)
            
            insert_position = self.test_df.shape[1] - 1  # This is the index of the last column
            df1_left = self.test_df.iloc[:, :insert_position]
            df1_last = self.test_df.iloc[:, insert_position:]
            self.test_df = pd.concat([df1_left, test_df_temp, df1_last], axis=1)
            
            self.train_df =  self.train_df.loc[:,~ self.train_df.columns.duplicated()].copy()
            self.validation_df = self.validation_df.loc[:,~ self.validation_df.columns.duplicated()].copy()
            self.test_df = self.test_df.loc[:,~ self.test_df.columns.duplicated()].copy()


# kwargs = {"arg1" : "Geeks", "arg2" : "for", "arg3" : "Geeks"}
# myFun(**kwargs)