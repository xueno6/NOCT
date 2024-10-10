# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:24:00 2024

@author: MSI-NB
"""
import os
# Content to be written in each Python file
file_content1 = """
import pandas as pd
import numpy as np
import time
import os
import pandas as pd
import numpy as np
import sys
# Modify sys.path to include the src directory
src_path = os.path.abspath('src')
sys.path.append(src_path)
from utility import set_starters, adjustTree,statistic_leafs_depth, Tee
from Instance import Instance
from OCTData import OCTData_quadratic, OCTData_aggressive
from OCTH import OCTHflow
from Trees import CustomDecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from tqdm import tqdm
from Convert import CART_warm_start, Var_to_FakeVar, SVMflow
from gurobipy import quicksum
from utility import node_cross
from Kernelfuncs import replace_feature, find_SVC
from Symbolicfuncs import replace_feature

sigma=3
beta = 0
mu = 1e-4
"""
file_content2= """
for iPartition in range(5):
    path = os.path.join('.', 'Partitions', folder, f"{folder}.txt")
    ins = Instance(path, False)
    ins.read()
    tr, va, te = ins.get_ith_partition(iPartition)
    oq = OCTData_aggressive(ins.df, tr, va, te)
    oq.insert([set([ind for ind in range(len(oq.train_df.columns)-1)])])
    # round 1: use original features
    with Tee("./featcorsssigma3Log/"+str(folder)+"_"+str(sigma)+"_"+str(iPartition)+'.log', 'w'):
        if (np.power(2, sigma)-1) * (len(oq.train_df.columns)-1)>2100:
            oq = replace_feature(oq)
        CMAX = (np.power(2, sigma)-1) * (len(oq.train_df.columns)-1)
        highest_acc = 0
        highest_records = []
        tree_records = []
        C = CMAX
        t1 = time.time()
        alpha = 0


        #             OCTHmodel = OCTHflow(0, beta, sigma, mu, sigma, oq.train_df)
        #             a, b, d, s, g, u = OCTHmodel.model(100, solve = True, starts=None)
        #             ObjVal = OCTHmodel.OCTH.ObjVal
        OCTHmodel = OCTHflow(alpha, beta, sigma, mu, C, oq.train_df)

        while C > sigma: 
            #alpha = ObjVal/(C+1)
            # set start
            at, bt, dt, st, gt = CART_warm_start(oq.train_df, sigma)
            a, b, d, s, g, u = OCTHmodel.model(120, solve = True, starts=None, C_new = C)
        
            # adjust tree
            aSVM, bSVM = adjustTree(a, b, d, g, s, u, OCTHmodel, oq)
            feature_pairs = node_cross(aSVM, d, OCTHmodel)
        
            # tree after adjust
            clf = CustomDecisionTreeClassifier(aSVM, bSVM, d, g, OCTHmodel)
            va_acc = clf.score(oq.validation_df[oq.validation_df.columns[:-1]], oq.validation_df['label'])
            # renew C
            c = quicksum(s[j,t].x for j in OCTHmodel.J for t in OCTHmodel.N)
            C = c.getValue()-1
            # optional (test error)
            test_acc = clf.score(oq.test_df[oq.test_df.columns[:-1]], oq.test_df['label'])
            # record real depth and number of leaf
            numberLeafs, maxDepth = statistic_leafs_depth(d, OCTHmodel)
            re = [OCTHmodel.OCTH.MIPGap, 
                  OCTHmodel.OCTH.ObjVal,
                  c.getValue(), 
                  va_acc,
                  test_acc,
                  numberLeafs,
                  maxDepth,
                  feature_pairs]
            print(re)
            if highest_acc <= va_acc:
                highest_acc = va_acc
                highest_records.append(re)
                tree_records.append(clf)
        t2 = time.time()
        total_time = t2 - t1
        print(total_time)
        df = pd.DataFrame(highest_records, columns=['Gap', 'Obj','C','validation_error', 'test_error', 'numberLeafs','maxDepth','fp'])
        sorted_df = df.sort_values(by=['validation_error', 'C'], ascending=[False, True])
        sorted_df.to_excel("./featcorsssigma3Log/"+str(folder)+"_"+str(sigma)+"_"+str(iPartition)+".xlsx", index=False)
        #sorted_highest_tree = tree_records[sorted_df.index[0]]
        #final_acc = sorted_highest_tree.score(oq.test_df[oq.test_df.columns[:-1]], oq.test_df['label'])
        final_fp = sorted_df['fp'].iloc[0]
        oq.insert2(final_fp)
    
    with Tee("./featcorss2thsigma3Log/"+str(folder)+"_"+str(sigma)+"_"+str(iPartition)+'.log', 'w'):
        if (np.power(2, sigma)-1) * (len(oq.train_df.columns)-1)>2100:
            oq = replace_feature(oq)
        CMAX = (np.power(2, sigma)-1) * (len(oq.train_df.columns)-1)
        highest_acc = 0
        highest_records = []
        tree_records = []
        C = CMAX
        t1 = time.time()
        alpha = 0
        OCTHmodel = OCTHflow(alpha, beta, sigma, mu, C, oq.train_df)
        #             OCTHmodel = OCTHflow(0, beta, sigma, mu, sigma, oq.train_df)
        #             a, b, d, s, g, u = OCTHmodel.model(100, solve = True, starts=None)
        #             ObjVal = OCTHmodel.OCTH.ObjVal

        while C > sigma: 
            #alpha = ObjVal/(C+1)

            # set start
            at, bt, dt, st, gt = CART_warm_start(oq.train_df, sigma)
            a, b, d, s, g, u = OCTHmodel.model(120, solve = True, starts=None, C_new = C)
        
            # adjust tree
            aSVM, bSVM = adjustTree(a, b, d, g, s, u, OCTHmodel, oq)
            feature_pairs = node_cross(aSVM, d, OCTHmodel)
        
            # tree after adjust
            clf = CustomDecisionTreeClassifier(aSVM, bSVM, d, g, OCTHmodel)
            va_acc = clf.score(oq.validation_df[oq.validation_df.columns[:-1]], oq.validation_df['label'])
            # renew C
            c = quicksum(s[j,t].x for j in OCTHmodel.J for t in OCTHmodel.N)
            C = c.getValue()-1
            # optional (test error)
            test_acc = clf.score(oq.test_df[oq.test_df.columns[:-1]], oq.test_df['label'])
            # record real depth and number of leaf
            numberLeafs, maxDepth = statistic_leafs_depth(d, OCTHmodel)
            re = [OCTHmodel.OCTH.MIPGap, 
                  OCTHmodel.OCTH.ObjVal,
                  c.getValue(), 
                  va_acc,
                  test_acc,
                  numberLeafs,
                  maxDepth,
                  feature_pairs]
            print(re)
            if highest_acc <= va_acc:
                highest_acc = va_acc
                highest_records.append(re)
                tree_records.append(clf)
        t2 = time.time()
        total_time = t2 - t1
        print(total_time)
        df = pd.DataFrame(highest_records, columns=['Gap', 'Obj','C','validation_error', 'test_error', 'numberLeafs','maxDepth','fp'])
        sorted_df = df.sort_values(by=['validation_error', 'C'], ascending=[False, True])
        sorted_df.to_excel("./featcorss2thsigma3Log/"+str(folder)+"_"+str(sigma)+"_"+str(iPartition)+".xlsx", index=False)
        #sorted_highest_tree = tree_records[sorted_df.index[0]]
        #final_acc = sorted_highest_tree.score(oq.test_df[oq.test_df.columns[:-1]], oq.test_df['label'])    
"""

# Loop to generate multiple Python files
folder_path = './Partitions'
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
for folder in subfolders:
    file_name = folder + "basicFunc.py"  # Generating the file name dynamically
    file_content_tmep = """folder = \'""" + folder + "\'"

    with open(file_name, 'w') as file:
        file.write(file_content1 + file_content_tmep + file_content2)  # Writing the predefined content to the file
        print(f"{file_name} has been created.")  # Optional: to confirm file creation
