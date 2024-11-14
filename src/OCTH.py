import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
def at(t):
    if t % 2 == 1:
        return int((t-1)/2)
    else:
        return int((t-2)/2)
def lt(t):
    return 2*t + 1
def rt(t):
    return 2*t + 2

def complete_tree_edges(n):
    edges = []
    for i in range(n):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child <= n:
            edges.append((i, left_child))
        if right_child <= n:
            edges.append((i, right_child))
    return edges

class OCTHflow:
    def __init__(self, alpha, beta, sigma, mu, C, df):
        self.alpha = alpha
        self.beta = beta
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
        self.OCTH = None
        countDict = {}
        for k in self.K:
            countDict[k] = 0
        for i in df.index:
            countDict[df['label'][i]] += 1
        self.Lhat = len(df)-max(countDict.values())

    def model(self, timelimit, solve = True, starts = None, C_new=None):
        if self.OCTH != None and C_new != None:
            c = self.OCTH.getConstrs()[-1]
            self.OCTH.setAttr(GRB.Attr.RHS, c, C_new)  # Set the new RHS value
            self.OCTH.update()
            if solve == True:
                self.OCTH.optimize()
                self.OCTH.update()
                return self.a, self.b, self.d, self.s, self.g, self.u
        s_node = -1
        w_node = self.L[-1]+1
        edges = complete_tree_edges(self.L[-1])
        edges = edges + [(-1, 0)]
        edges = edges + [(t, w_node) for t in self.N+self.L]
        
        self.OCTH = gp.Model()
        ahat = self.OCTH.addVars(self.J,self.N, vtype = GRB.CONTINUOUS, lb=0, ub =1)
        self.a = self.OCTH.addVars(self.J,self.N, vtype = GRB.CONTINUOUS, lb=-1, ub =1)
        self.b = self.OCTH.addVars(self.N, vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY)
        self.d = self.OCTH.addVars(self.N, vtype = GRB.BINARY)
        self.s = self.OCTH.addVars(self.J, self.N, vtype = GRB.BINARY)
        self.g = self.OCTH.addVars(self.K, self.N+self.L, vtype = GRB.BINARY)
        self.u = {}
        for i in self.I:
            for edge in edges:
                self.u[i, edge] = self.OCTH.addVar(vtype=GRB.BINARY)
        
        #  a, b, d, s, g start values
        if starts != None:
            for t in self.N:
                self.b[t].Start = starts['b'][t].x
                self.d[t].Start = starts['d'][t].x
            for t in self.N:
                for j in self.J:
                    self.a[j,t].Start = starts['a'][j,t].x
                    self.s[j,t].Start = starts['s'][j,t].x 
            
            for k in self.K:
                for t in self.N + self.L:
                    self.g[k,t].Start = starts['g'][k,t].x
        
        
        self.OCTH.addConstrs(quicksum(ahat[j,t] for j in self.J) <= self.d[t] for t in self.N)
        self.OCTH.addConstrs(-ahat[j,t]<= self.a[j,t] for j in self.J for t in self.N)
        self.OCTH.addConstrs(self.a[j,t] <= ahat[j,t] for j in self.J for t in self.N)
        self.OCTH.addConstrs(-self.s[j,t]<= self.a[j,t] for j in self.J for t in self.N)
        self.OCTH.addConstrs(self.a[j,t] <= self.s[j,t] for j in self.J for t in self.N)
        self.OCTH.addConstrs(self.s[j,t] <= self.d[t] for j in self.J for t in self.N)
        self.OCTH.addConstrs(quicksum(self.s[j,t] for j in self.J) >= self.d[t] for t in self.N)
        self.OCTH.addConstrs(-self.d[t] <= self.b[t] for t in self.N)
        self.OCTH.addConstrs(self.b[t] <= self.d[t] for t in self.N)
        self.OCTH.addConstrs(quicksum(self.g[k,l] for k in self.K) ==1 for l in self.L)
        self.OCTH.addConstrs(self.u[i,(at(t), t)] == 
                        self.u[i, (t, lt(t))] + self.u[i, (t, rt(t))] + self.u[i, (t, w_node)] for t in self.N[1:] for i in self.I)
        self.OCTH.addConstrs(self.u[i,(-1, 0)] == 
                        self.u[i, (0, lt(0))] + self.u[i, (0, rt(0))] + self.u[i, (0, w_node)] for i in self.I)
        self.OCTH.addConstrs(self.u[i, (at(l),l)] == self.u[i, (l, w_node)] for l in self.L for i in self.I)
        self.OCTH.addConstrs(self.u[i, (t, w_node)] <= self.g[self.df.iloc[i,-1], t] for i in self.I for t in self.N+self.L)
        self.OCTH.addConstrs(self.d[t] + quicksum(self.g[k,t] for k in self.K) == 1 for t in self.N)
        self.OCTH.addConstrs(quicksum(self.a[j,t]*self.df.iloc[i,j] for j in self.J) + self.mu <= 
                       self.b[t] + (2+self.mu)*(1-self.u[i, (t, lt(t))]) for t in self.N for i in self.I)
        self.OCTH.addConstrs(quicksum(self.a[j,t]*self.df.iloc[i,j] for j in self.J) >= 
                       self.b[t] - 2*(1-self.u[i,(t, rt(t))]) for t in self.N for i in self.I)
        self.OCTH.addConstrs(self.u[i, (t, lt(t))] <= self.d[t] for i in self.I for t in self.N)
        self.OCTH.addConstrs(self.u[i, (t, rt(t))] <= self.d[t] for i in self.I for t in self.N)
        self.OCTH.addConstr(quicksum(self.s[j,t] for j in self.J for t in self.N)<=self.C)
        
        self.OCTH.setObjective(1/self.Lhat *(len(self.I) - quicksum(self.u[i, (s_node, 0)] for i in self.I)) + 
                          self.alpha * quicksum(self.s[j,t] for j in self.J for t in self.N), GRB.MINIMIZE)
        self.OCTH.Params.TimeLimit = timelimit
        self.OCTH.Params.LogToConsole = 0
        self.OCTH.setParam('OutputFlag', 0)
        if solve == True:
            self.OCTH.optimize()
            self.OCTH.update()
            return self.a, self.b, self.d, self.s, self.g, self.u
    def extract():
        pass
