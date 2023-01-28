import numpy as np
import matplotlib.pyplot as plt

"""
n total number of samples
nHidden is the number of nodes in the hidden layer
f function used to evaluate
df f:s derivative
samples the samples as in 1a
T target values for respective sample
eta learing rate
"""
class TwoMLP:
    # constructor function
    def __init__(self, n, nHidden, f, df, samples, T, eta):
        self.n=n
        self.nHidden=nHidden
        self.f=np.vectorize(f)
        self.df=np.vectorize(df)
        self.samples = samples
        self.X = np.vstack((samples, np.array((n) * [1])))
        self.T=T
        self.eta=eta

    #computed for all samples
    def forward(self, W, V):
        #print("X ", self.X)
        hin=np.dot(W, self.X)
        #print("hin ", hin)
        hout=np.vstack((self.f(hin), self.n*[1]))
        #print("hout ", hout)
        oin=np.dot(V, hout)
        #print("oin ", oin)
        out=f(oin)
        #print("out ", out)
        return np.array((hin, hout, oin, out))

    #computed for all samples
    def backward(self, W, V):
        values=self.forward(W, V)
        hin, hout, oin, out=values[0], values[1], values[2], values[3]
        delta_o = np.multiply((out-self.T), self.df(oin))
        #print(np.outer(np.transpose(V), delta_o))
        delta_h = np.multiply(np.outer(np.transpose(V), delta_o), self.df(oin))
        #print(np.outer(np.transpose(V), delta_o), self.df(oin), delta_h)
        delta_h = delta_h[:-1]
        #print(delta_h)
        return np.array((delta_h, delta_o, hout))

    #batch
    def weightUpdate(self, W, V):
        values = self.backward(W, V)
        delta_h, delta_o, hout = values[0], values[1], values[2]
        delta_W = self.eta*np.dot(delta_h, np.transpose(self.X))
        delta_V = self.eta*np.dot(delta_o, np.transpose(hout))
        #print(delta_W, delta_V)
        #print(np.shape(delta_W), np.shape(delta_V))


'''
#Testing with three samples
def df(x):
    return x

def f(x):
    return x+10

samples=np.array([[1,2,3],[2,2,4]])
W=np.matrix([[1,2,3],[4,5,6]])
V=np.array([1,2,3])
T=np.array([1,0,2])

test=TwoMLP(3, 2, f, df, samples, T, 0.1)
test.weightUpdate(W, V)
'''
