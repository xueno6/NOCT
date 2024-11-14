# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:21:15 2024

@author: MSI-NB
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def replace_feature(oq):   
    target_column='label'
    
    model = SVC(C=1.0, kernel='linear')
    
    # Train the model
    model.fit(oq.train_df.drop(target_column, axis=1), oq.train_df[target_column])
    
    support_vectors = model.support_
    
    droped_cols = [item for ind, item in enumerate(oq.train_df.columns) if ind not in support_vectors and ind != len(oq.train_df.columns)-1]
    oq.train_df = oq.train_df.drop(columns=droped_cols)
    oq.validation_df = oq.validation_df.drop(columns=droped_cols)
    oq.test_df = oq.test_df.drop(columns=droped_cols)
    return oq