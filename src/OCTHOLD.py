import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
def pt(t):
    if t % 2 == 1:
        return int((t-1)/2)
    else:
        return int((t-2)/2)
def Alt(t):
    Al = []
    while 0 != t:
        if t == 2*pt(t)+1:
            Al.append(pt(t))
        t = pt(t) 
    return Al
def Art(t):
    Ar = []
    while 0 != t:
        if t == 2*pt(t)+2:
            Ar.append(pt(t))
        t = pt(t)
    return Ar

class OCTHnodes:
    def __init__(self, alpha, beta, sigma, mu, C, df):
        self.alpha = alpha
        self.beta = 0.05 * len(df) 
        self.sigma = sigma
        #self.Lhat = Lhat
        self.mu = mu
        self.C = C
        self.df = df
        self.K = set(df['label'])
        self.I = [i for i in range(len(df))]
        self.J = [j for j in range(len(df.columns)-1)]
        self.N = [t for t in range(np.power(2, sigma)-1)]
        self.L = [l for l in range(len(self.N), len(self.N)+np.power(2, sigma))]
        
        countDict = {}
        for k in self.K:
            countDict[k] = 0
        for i in df.index:
            countDict[df['label'][i]] += 1
        self.Lhat = len(df)-max(countDict.values())

    def model(self, timelimit, solve = True, starts = None):
        n = len(self.I)
        Y = {}
        for i in self.I:
            for k in self.K:
                if self.df.iloc[i, -1] == k:
                    Y[i,k] = 1
                else:
                    Y[i,k] = -1
                    
        
        OCTH = gp.Model()
        ahat = OCTH.addVars(self.J,self.N, vtype = GRB.CONTINUOUS, lb=0, ub =1)
        a = OCTH.addVars(self.J,self.N, vtype = GRB.CONTINUOUS, lb=-1, ub =1)
        b = OCTH.addVars(self.N, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY)
        d = OCTH.addVars(self.N, vtype = GRB.BINARY)
        s = OCTH.addVars(self.J, self.N, vtype = GRB.BINARY)
        z = OCTH.addVars(self.I, self.N + self.L, vtype = GRB.BINARY)
        c = OCTH.addVars(self.K, self.L, vtype = GRB.BINARY)
        L = OCTH.addVars(self.L, vtype = GRB.CONTINUOUS)
        N1 = OCTH.addVars(self.L, vtype = GRB.CONTINUOUS)
        N2 = OCTH.addVars(self.K, self.L, vtype = GRB.CONTINUOUS)
        l = OCTH.addVars(self.L, vtype = GRB.BINARY)
        
        #  a, b, d, s, g start values
        if starts != None:
            for t in self.N:
                b[t].Start = starts['b'][t].x
                d[t].Start = starts['d'][t].x
            for t in self.N:
                for j in self.J:
                    a[j,t].Start = starts['a'][j,t].x
                    s[j,t].Start = starts['s'][j,t].x 
            
            for k in self.K:
                for t in self.L:
                    c[k,t].Start = starts['g'][k,t].x
        
        OCTH.addConstrs(L[t] >= N1[t] - N2[k,t] - n*(1-c[k,t]) for k in self.K for t in self.L)
        OCTH.addConstrs(L[t] <= N1[t] - N2[k,t] + n*c[k,t] for k in self.K for t in self.L)
        OCTH.addConstrs(N2[k,t] == 1/2* quicksum( (1+Y[i,k])*z[i,t] for i in self.I) for k in self.K for t in self.L)
        OCTH.addConstrs(N1[t] == quicksum(z[i,t] for i in self.I) for t in self.L)
        OCTH.addConstrs(quicksum(c[k,t] for k in self.K) == l[t] for t in self.L)
        OCTH.addConstrs(quicksum(a[j,m]*self.df.iloc[i,j] for j in self.J) + self.mu <= 
                       b[m] + (2+self.mu)*(1-z[i, t]) for t in self.L for i in self.I for m in Alt(t))
        OCTH.addConstrs(quicksum(a[j,m]*self.df.iloc[i,j] for j in self.J) >= 
                       b[m] - 2*(1-z[i,t]) for t in self.L for i in self.I for m in Art(t))
        OCTH.addConstrs(quicksum(z[i,t] for t in self.L) == 1 for i in self.I)
        OCTH.addConstrs(z[i,t] <= l[t] for t in self.L for i in self.I)
        OCTH.addConstrs(quicksum(z[i,t] for i in self.I) >= self.beta * l[t] for t in self.L)
        OCTH.addConstrs(quicksum(ahat[j,t] for j in self.J) <= d[t] for t in self.N)
        OCTH.addConstrs(-ahat[j,t]<= a[j,t] for j in self.J for t in self.N)
        OCTH.addConstrs(a[j,t] <= ahat[j,t] for j in self.J for t in self.N)
        OCTH.addConstrs(-s[j,t]<= a[j,t] for j in self.J for t in self.N)
        OCTH.addConstrs(a[j,t] <= s[j,t] for j in self.J for t in self.N)
        OCTH.addConstrs(s[j,t] <= d[t] for j in self.J for t in self.N)
        OCTH.addConstrs(quicksum(s[j,t] for j in self.J) >= d[t] for t in self.N)
        OCTH.addConstrs(-d[t] <= b[t] for t in self.N)
        OCTH.addConstrs(b[t] <= d[t] for t in self.N)
        OCTH.addConstrs(d[t] <= d[pt(t)] for t in self.N if t != 0)
        OCTH.addConstr(quicksum(s[j,t] for j in self.J for t in self.N)<=self.C)
        
        OCTH.setObjective(1/self.Lhat *quicksum(L[t] for t in self.L) + 
                          self.alpha * quicksum(s[j,t] for j in self.J for t in self.N), GRB.MINIMIZE)
        OCTH.Params.TimeLimit = timelimit
        #OCTH.Params.LogToConsole = 0
        self.OCTH = OCTH
        if solve == True:
            self.OCTH.optimize()
            self.OCTH.update()
            return a, b, d, l, c
    def extract():
        pass
