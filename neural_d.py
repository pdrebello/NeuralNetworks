import numpy as np
from scipy.special import expit

"""def softmax(X, theta = 1.0, axis = 1):
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p"""

def softmax(X):
    a = np.exp(X)
    b = np.sum(a,axis=1).reshape((X.shape[0],1))
    return a/b
p = 0
lamb = 0.05
class Layer:
    def __init__(self, inputsize, outputsize,activation='sigmoid',lastlayer=False):
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.activation = activation
        self.lastlayer = lastlayer
        self.weights = np.random.randn(inputsize+1,outputsize)*np.sqrt(1/(inputsize))
        self.bias = np.random.randn((outputsize))*np.sqrt(1/(inputsize))
        self.dropout = np.ones((outputsize,1))
        #self.bias = np.zeros((outputsize),dtype=np.float64)
        #self.weights = np.zeros((inputsize+1,outputsize),dtype=np.float64)
    def forwardprop(self, inp):
        self.input = inp
        self.dropout = np.random.choice(a=[False, True], size=((self.outputsize)), p=[p, 1-p])
        self.dropout = np.array([self.dropout,]*self.input.shape[0])
        self.output = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1),self.weights)
        if(self.lastlayer == False):
            self.output = self.output*self.dropout*(1/(1-p))
        if(self.activation == 'softmax'):
            self.output = softmax(self.output)
        elif(self.activation == 'relu'):
            self.output = np.array(self.output>0,dtype=int)*self.output
        else:
            self.output = expit(self.output)
        return self.output
    
    def backprop(self, derz, eta):
        if(self.lastlayer == True):
            self.derivative = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1).T,derz)
            self.derz = np.dot(derz,self.weights[1:,:].T)
        elif(self.activation == 'relu'):
            self.derivative = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1).T,derz*np.array(self.output>0,dtype=int))            
            self.derz = np.dot(derz*np.array(self.output>0,dtype=int),self.weights[1:,:].T)
        else:
            self.derivative = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1).T,self.output*(1-self.output)*derz)            
            self.derz = np.dot(self.output*(1-self.output)*derz,self.weights[1:,:].T)
            #self.derivativeb = (self.output*(1-self.output)*derz)
            
        self.weights = self.weights - eta*self.derivative -(lamb*self.weights)/self.input.shape[0]
        return self.derz

class NeuralNetwork:
    def __init__(self,inputsize):
        self.inputsize = inputsize
        self.LayerList = []

    def addLayer(self,outputsize,activation='sigmoid'):
        if(len(self.LayerList) == 0):
            lay = Layer(self.inputsize,outputsize,activation,True)
            self.LayerList.append(lay)
        else:
            self.LayerList[len(self.LayerList)-1].lastlayer=False
            lay = Layer(self.LayerList[len(self.LayerList)-1].outputsize,outputsize,activation,True)
            self.LayerList.append(lay)

    def forwardprop(self, inp):
        for i in range(len(self.LayerList)):
            inp = self.LayerList[i].forwardprop(inp)
        return inp
    
    def backprop(self, inp, Y,eta):
        output = self.forwardprop(inp)
        derz = (output - Y)/Y.shape[0]
        for i in range(len(self.LayerList)):
            derz = self.LayerList[len(self.LayerList)-i-1].backprop(derz,eta)
        return derz
    def loss(self,X,Y):
        output = self.forwardprop(X)
        if(self.LayerList[len(self.LayerList)-1].activation=='softmax'):
            output = np.log(output)
            loss = output*Y
            summ = np.sum(np.sum(loss))
            return -summ/X.shape[0] 
        else:
            diff = Y*np.log(output+1e-10) + (1-Y)*(1-np.log(output+1e-10))
            return -np.sum(diff)/X.shape[0]
    def accuracy(self,X,Y):
        Ycap = network.forwardprop(X)
        a = np.argmax(Ycap,axis=1)
        b = np.argmax(Y,axis=1)
        count=0
        for i in range(Y.shape[0]):
            if(a[i]==b[i]):
                count+=1
        return float(count)/Y.shape[0]    
    
def MiniBatchGradientDescent(X,Y,network,epochs,minibatchsize,eta,Xtest,Ytest,adaptive=1):
    if(adaptive==1):
        for i in range(epochs):
            for j in range(int(X.shape[0]/minibatchsize)):
                network.backprop(X[j*minibatchsize:(j+1)*minibatchsize],Y[j*minibatchsize:(j+1)*minibatchsize],eta)
            #print(network.loss(X,Y))
                
    else:
        for i in range(epochs):
            for j in range(int(X.shape[0]/minibatchsize)):
                #print(str(i) +" " + str(j))
                network.backprop(X[j*minibatchsize:(j+1)*minibatchsize],Y[j*minibatchsize:(j+1)*minibatchsize],eta)
                #if(j%10 == 0):
                #if(j%100 == 0):
            if(i%10==0):
                print(str(i))
                #print(i)
                    #print()
            eta = (eta*np.sqrt(i+1))/np.sqrt(i+2)
            
    return network

def OneHotEncoding(Y,classsize):
    Ypri = np.zeros((Y.shape[0],classsize))
    for i in range(Y.shape[0]):
        Ypri[i][int(Y[i])] = 1
    return Ypri

import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

#arguments = ['Neural_data/CIFAR10/train.csv','Neural_data/CIFAR10/param.txt','weightfile.txt']
trainfilename = arguments[0]
testfilename = arguments[1]
outputfilename = arguments[2]




from skimage import filters
from skimage import feature
#X = X/255.0
X = np.loadtxt(trainfilename,delimiter=',',dtype=np.float64)
Y = X[:,1024].reshape((20000,1))
Y = OneHotEncoding(Y,10).reshape(20000,10)
X = X[:,0:1024]
X = X/255.0
me = np.mean(X)
X = X - me


Xtrainpri = X.reshape((X.shape[0],32,32))
Xtrainpri = np.flip(Xtrainpri,axis=2)
Xtrainpri = Xtrainpri.reshape((X.shape[0],1024))
X = np.concatenate((X,Xtrainpri),axis=0)
Y = np.concatenate((Y,Y),axis=0)

#from random import shuffle
#concat = np.concatenate((X,Y),axis=1)
#shuffle(concat)
#X = concat[:,0:1024]
#Y = concat[:,1024:].reshape((X.shape[0],10))

Xwewant = X.reshape((X.shape[0],32,32))

Xsobel = np.zeros((X.shape[0],32,32))
for i in range(X.shape[0]):
    Xsobel[i] = filters.sobel(Xwewant[i])
Xsobel = Xsobel.reshape((X.shape[0],1024))



"""Xcanny = np.zeros((36000,32,32))
for i in range(36000):
    Xcanny[i] = feature.canny(Xwewant[i])
Xcanny = Xcanny.reshape((36000,1024))"""

X = np.concatenate((X,Xsobel),axis=1)


network = NeuralNetwork(X.shape[1])
network.addLayer(200,'relu')
network.addLayer(10,'softmax')


network = MiniBatchGradientDescent(X,Y,network,80,200,0.5,Xtest=None,Ytest=None,adaptive=2)




Xtesting = np.loadtxt(testfilename,delimiter=',',dtype=np.float64)
Xtesting = Xtesting[:,0:1024]
Xtesting = Xtesting/255.0
Xtesting = Xtesting - me


Xwewant = Xtesting.reshape((Xtesting.shape[0],32,32))

Xsobel = np.zeros((Xtesting.shape[0],32,32))
for i in range(Xtesting.shape[0]):
    Xsobel[i] = filters.sobel(Xwewant[i])
Xsobel = Xsobel.reshape((Xtesting.shape[0],1024))
Xtesting = np.concatenate((Xtesting,Xsobel),axis=1)




Ycap = network.forwardprop(Xtesting)
Ycap = np.argmax(Ycap,axis=1)

with open(outputfilename,"w") as file:
    for i in range(Ycap.shape[0]):
        file.write(str(Ycap[i]))
        file.write("\n")
        