# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:26:48 2024

@author: MSI-NB
"""
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def replace_feature(oq, model, gamma, clusters=None):
    target_column='label'
    if clusters == None:
        support_vectors = model.support_
        colTemp = oq.train_df.iloc[support_vectors].drop(target_column, axis=1)
    else:
        colTemp = clusters
    y = oq.train_df[target_column]
    rowTemp = oq.train_df.drop(target_column, axis=1)
    # Compute the RBF kernel between each instance in X and the support vectors
    transformed_features = rbf_kernel(rowTemp, colTemp, gamma=gamma)
    # Create a new DataFrame for the transformed features
    oq.train_df = pd.DataFrame(transformed_features, 
                                  index=oq.train_df.index,
                                  columns=[f'kernel_{i}' for i in range(transformed_features.shape[1])])
    # Add the target column back
    oq.train_df[target_column] = y
    
    y = oq.validation_df[target_column]
    rowTemp = oq.validation_df.drop(target_column, axis=1)
    transformed_features = rbf_kernel(rowTemp, colTemp,  gamma=gamma)
    # Create a new DataFrame for the transformed features
    oq.validation_df = pd.DataFrame(transformed_features, 
                                  index=oq.validation_df.index,
                                  columns=[f'kernel_{i}' for i in range(transformed_features.shape[1])])
    oq.validation_df[target_column] = y
    
    y = oq.test_df[target_column]
    rowTemp = oq.test_df.drop(target_column, axis=1)
    transformed_features = rbf_kernel(rowTemp, colTemp,  gamma=gamma)
    # Create a new DataFrame for the transformed features
    oq.test_df = pd.DataFrame(transformed_features, 
                                  index=oq.test_df.index,
                                  columns=[f'kernel_{i}' for i in range(transformed_features.shape[1])])
    oq.test_df[target_column] = y
    return oq

def find_SVC(oq):
    
    target_column='label'
    X = oq.train_df.drop(target_column, axis=1)
    y = oq.train_df[target_column]
    # Define a range of C values to explore
    param_grid = {'C': [100, 500, 1000, 5000, 10000]}

    # Create a GridSearchCV object
    svc = SVC(kernel='rbf', gamma='scale')
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')  # 5-fold cross-validation

    # Fit GridSearchCV
    grid_search.fit(X, y)

    # Best C value
    best_C = grid_search.best_params_['C']
    #print(f"Best C Parameter: {best_C}")

    # Optionally, evaluate on the validation set
    best_model = grid_search.best_estimator_
    va_acc = best_model.score(oq.validation_df[oq.validation_df.columns[:-1]], oq.validation_df['label'])
    test_acc = best_model.score(oq.test_df[oq.test_df.columns[:-1]], oq.test_df['label'])
    gamma = 1 / (X.shape[1] * X.var().mean())
    return best_model, va_acc, test_acc, gamma
