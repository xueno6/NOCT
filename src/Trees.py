# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:50:45 2024

@author: MSI-NB
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from gurobipy import quicksum
class CustomDecisionTreeNode:
    def __init__(self, split_func=None, left=None, right=None, value=None):
        self.split_func = split_func  # Function to determine the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class value for leaf nodes
        if value != None and split_func != None:
            raise ValueError("A node must be either split node or leaf")
    def is_leaf(self):
        return self.value is not None  # Returns True if the node is a leaf

class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, a, b, d, g, OCTHmodel):
        self.is_fitted_ = True  # This model does not require training
        self.tree_ = self._build_tree(a, b, d, g, OCTHmodel)  # Build the tree structure
    def _build_tree(self, a, b, d, g, OCTHmodel):
        # Create nodes and assign split functions from the list
        nodes = {}
        for t in OCTHmodel.N:
            if d[t].x >0.5:
                nodes[t] = CustomDecisionTreeNode(split_func= create_function(a, b, t, OCTHmodel.J))
            else:
                k = [index for index in OCTHmodel.K if g[index, t].x>0.5]
                nodes[t] =  CustomDecisionTreeNode(value= k[0])
        for t in OCTHmodel.L:
            k = [index for index in OCTHmodel.K if g[index, t].x>0.5]
            nodes[t] =  CustomDecisionTreeNode(value= k[0])
        
        for t in OCTHmodel.N:
            nodes[t].left = nodes[2*t + 1]
            nodes[t].right = nodes[2*t + 2]
        return nodes[0]

    def fit(self, X=0, y=0):
        pass
        # This model does not require fitting, so we just return self
        # self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Apply the predefined rules to classify each sample
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def _predict_single(self, x):
        node = self.tree_
        while not node.is_leaf():
            if node.split_func(x):
                node = node.right
            else:
                node = node.left
        return node.value

class CustomNodesDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, a, b, d, g, OCTHmodel):
        self.is_fitted_ = True  # This model does not require training
        self.tree_ = self._build_tree(a, b, d, g, OCTHmodel)  # Build the tree structure
    def _build_tree(self, a, b, d, l, c, OCTHmodel):
        # Create nodes and assign split functions from the list
        nodes = {}
        for t in OCTHmodel.N:
            if d[t].x >0.5:
                nodes[t] = CustomDecisionTreeNode(split_func= create_function(a, b, t, OCTHmodel.J))
            else:
                nodes[t] =  CustomDecisionTreeNode(value= -1)
        for t in OCTHmodel.L:
            if l[t].x>0.5:
                k = [index for index in OCTHmodel.K if c[index, t].x>0.5]
                nodes[t] =  CustomDecisionTreeNode(value= k[0])
            else:
                pass
        for t in OCTHmodel.L:
            if l[t].x>0.5:
                pass
            else:
                at = (t-1) // 2 
                if 1 == t%2:
                    nodes[at].value = nodes[t+1].value
                    nodes[t] = CustomDecisionTreeNode(value= nodes[t+1].value)
                else:
                    nodes[at].value = nodes[t-1].value
                    nodes[t] = CustomDecisionTreeNode(value= nodes[t-1].value)

        for t in OCTHmodel.N:
            nodes[t].left = nodes[2*t + 1]
            nodes[t].right = nodes[2*t + 2]
        return nodes[0]

    def fit(self, X=0, y=0):
        pass
        # This model does not require fitting, so we just return self
        # self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Apply the predefined rules to classify each sample
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def _predict_single(self, x):
        node = self.tree_
        while not node.is_leaf():
            if node.split_func(x):
                node = node.right
            else:
                node = node.left
        return node.value

def create_function(a, b, t, J):
    indice = [index for index in J if abs(a[index, t].x)>1e-6]
    a_val = [a[j,t].x for j in J ]
    b_val = b[t].x
    def generic_function(X):
        c = quicksum(a_val[i]*X[i] for i in indice)
        return c.getValue()>b_val
    return generic_function