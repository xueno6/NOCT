# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:48:20 2024

@author: MSI-NB
"""
import gurobipy as gp
from gurobipy import GRB, quicksum
from sklearn.tree import DecisionTreeClassifier
from Trees import CustomDecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

class FakeVar:
    def __init__(self, value):
        self.x = value

class NodeSVM:
    def __init__(self, a, b, d, s, g, u, OCTHmodel):
        self.a = a
        self.b = b
        self.d = d
        self.s = s
        self.g = g
        self.u = u
        self.OCTHmodel = OCTHmodel
    def compute(self, t):
        IL = []
        IR = []
        for i in self.OCTHmodel.I:
            if self.u[i, (t, 2*t+1)].x>=0.5:
                IL.append(i)
            elif self.u[i,(t, 2*t+2)].x>=0.5:
                IR.append(i)
                
        NodeSVM = gp.Model()
        e = NodeSVM.addVars(IL+IR, vtype = GRB.CONTINUOUS)
        em = NodeSVM.addVar(vtype=GRB.CONTINUOUS)
        st = NodeSVM.addVars(self.OCTHmodel.J, vtype = GRB.BINARY)
        at = NodeSVM.addVars(self.OCTHmodel.J, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY)
        bt = NodeSVM.addVar(vtype=GRB.CONTINUOUS, lb  = -1, ub = 1)
        
        NodeSVM.addConstrs(e[i] >= em for i in IL + IR)
        NodeSVM.addConstrs(e[i] == bt - quicksum(at[j]*self.OCTHmodel.df.iloc[i,j] for j in self.OCTHmodel.J) for i in IL)
        NodeSVM.addConstrs(e[i] == -bt + quicksum(at[j]*self.OCTHmodel.df.iloc[i,j] for j in self.OCTHmodel.J) for i in IR)
        NodeSVM.addConstrs(-st[j] <= at[j] for j in self.OCTHmodel.J)
        NodeSVM.addConstrs(at[j] <= st[j] for j in self.OCTHmodel.J)
        NodeSVM.addConstr(quicksum(st[j] for j in self.OCTHmodel.J) <= quicksum(self.s[j, t].x for j in self.OCTHmodel.J))
        NodeSVM.setObjective(em, GRB.MAXIMIZE)
        NodeSVM.Params.LogToConsole = 0
        NodeSVM.Params.TimeLimit = 10

        NodeSVM.optimize()
        tempa = {}
        for j in self.OCTHmodel.J:
            tempa[j, t] = FakeVar(at[j].x)
        tempb = {}
        tempb[t] = FakeVar(bt.x)
    
        return tempa, tempb


def SVMflow(a, b, d, s, g, u, OCTHmodel, oq, va_acc):
    aSVM = {}
    aSVMt = {}
    bSVM = {}
    bSVMt = {}
    for key in a.keys():
        aSVM[key] = FakeVar(a[key].x)
        aSVMt[key] = FakeVar(a[key].x)
    for key in b.keys():
        bSVM[key] = FakeVar(b[key].x)
        bSVMt[key] = FakeVar(b[key].x)
    
    NS = NodeSVM(a, b, d, s, g, u, OCTHmodel)
    for t in OCTHmodel.N:
        if d[t].x > 0.5:
            tempa, tempb = NS.compute(t)
            for key in tempa.keys():
                aSVMt[key] = tempa[key]
            for key in tempb.keys():
                bSVMt[key] = tempb[key]
            clf = CustomDecisionTreeClassifier(aSVMt, bSVMt, d, g, OCTHmodel)
            result = clf.predict(oq.validation_df[oq.validation_df.columns[:-1]])
            va_acc_temp = accuracy_score(result, oq.validation_df['label'])
            #print(va_acc_temp)
            if va_acc_temp > va_acc:
                for key in tempa.keys():
                    aSVM[key] = tempa[key]
                for key in tempb.keys():
                    bSVM[key] = tempb[key]
            else:
                for key in tempa.keys():
                    aSVMt[key] = aSVM[key]
                for key in tempb.keys():
                    bSVMt[key] = bSVM[key]       
    return aSVM, bSVM


def find_parent(tree, node_id):
    # The root node has no parent
    if node_id == 0:
        return None 
    # Iterate over all nodes to find the parent
    for i in range(node_id):
        if tree.children_left[i] == node_id:
            return i, 1
        elif tree.children_right[i] == node_id:
            return i, 2
    return None  # If no parent is found (which is unusual)

def CART_warm_start(df, sigma):
    classifier = DecisionTreeClassifier(max_depth=sigma, random_state=42)
    X = df.drop('label', axis=1)
    y = df['label']
    classifier.fit(X, y)
    tree = classifier.tree_
    
    K = set(df['label'])
    I = [i for i in range(len(df))]
    J = [j for j in range(len(df.columns)-1)]
    N = [t for t in range(np.power(2, sigma)-1)]
    L = [l for l in range(len(N), len(N)+np.power(2, sigma))]
    
    a, b, d, s, g = {}, {}, {}, {}, {}
    for t in N:
        b[t] = FakeVar(0)
        d[t] = FakeVar(0)
    for t in N:
        for j in J:
            a[j,t] = FakeVar(0)
            s[j,t] = FakeVar(0)
    
    for k in K:
        for t in N + L:
            g[k,t] = FakeVar(0)
    i_to_t_dict = {}
    i_to_t_dict[0] = 0
    for i in range(1, tree.node_count):
        parent, postfix = find_parent(tree, i)
        i_to_t_dict[i] = 2* i_to_t_dict[parent] + postfix 
    
    for i in range(tree.node_count):
        t = i_to_t_dict[i]
        if tree.children_left[i] != tree.children_right[i]:  # Only non-leaf nodes
            #print(f"{i} | {oq.train_df.columns[tree.feature[i]]} | {tree.threshold[i]}")
            d[t] =  FakeVar(1)
            b[t] = FakeVar(tree.threshold[i])
            a[tree.feature[i],t] = FakeVar(1)
            s[tree.feature[i],t] = FakeVar(1)
        else:
            #print(f"{i} is a leaf node.{tree.value[i][0]}")
            k = [ind for ind, item in enumerate(tree.value[i][0]) if item!=0][0]
            g[k,t] = FakeVar(1)
    for t in N+L:
        if t not in i_to_t_dict.values():
            #print(t)
            parent = (t-0.5)//2
            k = [kk for kk in K if g[kk, parent].x >0.5][0]
            g[k,t] = FakeVar(1)
    return a, b, d, s, g
    
    
    
def Var_to_FakeVar(var):
    var_dict = {}
    for key in var.keys():
        var_dict[key] = FakeVar(var[key].x)
    return var_dict
    
    
    
    
