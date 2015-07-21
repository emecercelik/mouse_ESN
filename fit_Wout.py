import pickle
import numpy.random as random
import math as m
from numpy import linalg as LA
import numpy as np
# =================================================================================================================================================
#                                       Functions

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def ActFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def NeuronFunc(x):
    return (1.+np.tanh(x-33.))*.5 # 4.=9 input

def InputFunc(x,dim,nAct,maxAcc,minAcc):
    rangeAcc=maxAcc-minAcc
    off=np.arange(minAcc+rangeAcc/(2.*nAct),maxAcc,rangeAcc/nAct)
    res=np.array([[ActFunc(nAct,maxAcc,minAcc,x[j],off[i]) for i in range(len(off))] for j in range(len(x))])
    res=res.reshape(res.size,1)
    for i in range(res.size):
        if res[i]<1e-2:
            res[i]=.0
    return res


numOut=6
##a=np.vstack((GetPickle('paramWalking1'),GetPickle('paramWalking1')))
##a=np.vstack((GetPickle('paramWalking1'),GetPickle('paramWalking2'),\
##             GetPickle('paramWalking3'),GetPickle('paramWalking4'),\
##             GetPickle('paramWalking5'),GetPickle('paramWalking6')))
a=GetPickle('paramWalking1')
b=a[2:,:-numOut]
c=a[2:,-numOut:]

wout=c.T.dot(b)
inverse=LA.inv(b.T.dot(b)+.01*np.random.rand(b.shape[1],b.shape[1]))+1e-8*np.identity(int(b.shape[1]))
wout=wout.dot(inverse)
PickleIt(wout,'wout')

