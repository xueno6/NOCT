# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:59:11 2024

@author: MSI-NB
"""

import gurobipy as gp
from gurobipy import GRB, quicksum
import os
from Convert import CART_warm_start, Var_to_FakeVar, SVMflow
from Trees import CustomDecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import sys

def node_cross(a, d, OCTHmodel):
    feature_pairs = []
    for t in OCTHmodel.N:
        if d[t].x>0.5:
            feature_to_cross = set([index for index in OCTHmodel.J if abs(a[index, t].x)>1e-6])
            drop_indices = []
            contain = False
            for ind, sub in enumerate(feature_pairs):
                if sub >= feature_to_cross:
                    contain = True
                    break
                elif feature_to_cross >= sub:
                    drop_indices.append(ind)
            if contain == False:
                feature_pairs = [fp for ind, fp in enumerate(feature_pairs) if ind not in drop_indices]
                feature_pairs.append(feature_to_cross)
    return feature_pairs


def list_subfolders(folder_path):
    try:
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        return subfolders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def set_starters(a, b, d, s, g):
    starts = {}
    starts['a'] = Var_to_FakeVar(a)
    starts['b'] = Var_to_FakeVar(b)
    starts['d'] = Var_to_FakeVar(d)
    starts['s'] = Var_to_FakeVar(s)
    starts['g'] = Var_to_FakeVar(g)
    return starts

def adjustTree(a, b, d, g, s, u, OCTHmodel, oq):
    # implement to a tree and validate
    clf = CustomDecisionTreeClassifier(a, b, d, g, OCTHmodel)
    result = clf.predict(oq.validation_df[oq.validation_df.columns[:-1]])
    va_acc = accuracy_score(result, oq.validation_df['label'])
    # adjust algorithm
    aSVM, bSVM = SVMflow(a, b, d, s, g, u, OCTHmodel, oq, va_acc)
    return aSVM, bSVM

def statistic_leafs_depth(d, OCTHmodel):
    leafs = [1 for i in OCTHmodel.N + OCTHmodel.L]
    delete_ind = []
    for i in OCTHmodel.N:
        if d[i].x>0.5:
            leafs[i] = 0
        else:
            delete_ind.append(2*i+1)
            delete_ind.append(2*i+2)
    numberLeafs = sum([i for ind,i in enumerate(leafs) if ind not in delete_ind])
    maxDepth = max([(1-ind)*i for ind,i in zip (leafs, OCTHmodel.N+OCTHmodel.L)])
    maxDepth = int(np.log(maxDepth+1)/np.log(2))+ 1
    return numberLeafs, maxDepth

class Tee:
    """A custom class to write output simultaneously to stdout and a file."""
    def __init__(self, file_name, mode):
        self.file = open(file_name, mode)
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        # Flush both stdout and the file
        self.stdout.flush()
        self.file.flush()