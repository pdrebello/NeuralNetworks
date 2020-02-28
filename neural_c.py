import numpy as np
from scipy.special import expit

def softmax(X, theta = 1.0, axis = 1):
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

    return p

class Layer:
    def __init__(self, inputsize, outputsize,activation='sigmoid',lastlayer=False):
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.activation = activation
        self.lastlayer = lastlayer
        #self.weights = np.random.rand(inputsize+1,outputsize)*np.sqrt(2/(inputsize))
        #self.bias = np.random.rand((outputsize))*np.sqrt(2/(inputsize))
        self.bias = np.zeros((outputsize))
        self.weights = np.zeros((inputsize+1,outputsize))
    def forwardprop(self, inp):
        self.input = inp
        self.output = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1),self.weights)
        
        if(self.activation == 'softmax'):
            self.output = softmax(self.output)
        else:
            self.output = expit(self.output)
        return self.output
    
    def backprop(self, derz, eta):
        #if(self.lastlayer == True):
        #    self.derivative = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1).T,derz)  
        #    self.derz = np.dot(derz,self.weights[1:,:].T)
            #self.derivativeb = derz
            #print(self.derivativeb.shape)
            #print(self.bias.shape)
        #else:
       # a = self.output*(1-self.output)
    #    a = derz*a
        self.derivative = np.dot(np.concatenate((np.ones((self.input.shape[0],1)),self.input),axis=1).T,self.output*(1-self.output)*derz)            
        self.derz = np.dot(self.output*(1-self.output)*derz,self.weights[1:,:].T)
            #self.derivativeb = (self.output*(1-self.output)*derz)
            
        self.weights = self.weights - eta*self.derivative
        #self.bias = self.bias - eta*np.sum(self.derivativeb)
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
        #derz = (output - Y)/Y.shape[0]
        derz = ((1-Y)/(1-output)-(Y/output))/Y.shape[0]
        for i in range(len(self.LayerList)):
            derz = self.LayerList[len(self.LayerList)-i-1].backprop(derz,eta)
        return derz
    def loss(self,X,Y):
        output = self.forwardprop(X)
        if(self.LayerList[len(self.LayerList)-1].activation=='softmax'):
            output = np.log(output+1e-9)
            loss = output*Y
            summ = np.sum(np.sum(loss))
            return -summ/X.shape[0] 
        else:
            diff = Y*np.log(output+1e-10) + (1-Y)*(1-np.log(output+1e-10))
            return -np.sum(diff)/X.shape[0]
    def accuracy(self,X,Y):
        Ycap = network.forwardprop(X)
        count=0
        for i in range(Y.shape[0]):
            if(Y[i]==0):
                if(Ycap[i] < 0.5):
                    count+=1
            else:
                if(Ycap[i]>0.5):
                    count+=1
        return float(count)/Y.shape[0]
    
    
def MiniBatchGradientDescent(X,Y,network,max_iterations,minibatchsize,eta,adaptive=1):
    count = 0
    if(adaptive==1):
        for i in range(max_iterations):
            for j in range(int(X.shape[0]/minibatchsize)):
                if(count == max_iterations):
                    return network
                network.backprop(X[j*minibatchsize:(j+1)*minibatchsize],Y[j*minibatchsize:(j+1)*minibatchsize],eta)
                count+=1
    else:
        for i in range(max_iterations):
            for j in range(int(X.shape[0]/minibatchsize)):
                etan = eta/np.sqrt(count+1)
                if(count == max_iterations):
                    return network
                network.backprop(X[j*minibatchsize:(j+1)*minibatchsize],Y[j*minibatchsize:(j+1)*minibatchsize],etan)
                count+=1
    return network
    
def OneHotEncoding(Y,classsize):
    Ypri = np.zeros((Y.shape[0],classsize))
    for i in range(Y.shape[0]):
        Ypri[i][int(Y[i])] = 1
    return Ypri

import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

#arguments = ['Neural_data/Toy/train.csv','Neural_data/Toy/param.txt','weightfile.txt']
trainfilename = arguments[0]
paramfilename = arguments[1]
weightfilename = arguments[2]


f = open(paramfilename,"r")
f1 = f.readlines()
mode = int(f1[0].strip())
etan = float(f1[1].strip())
max_iterations = int(f1[2].strip())
batchsize = int(f1[3].strip())
networklayer = (f1[4].strip())
networklayer = networklayer.split(" ")


X = np.loadtxt(trainfilename,delimiter=',',dtype=np.float)
Y = X[:,X.shape[1]-1].reshape((X.shape[0],1))

X = X[:,0:X.shape[1]-1]

network = NeuralNetwork(X.shape[1])
for i in networklayer:
    network.addLayer(int(i))
network.addLayer(1)

network = MiniBatchGradientDescent(X,Y,network,max_iterations,batchsize,etan,adaptive=mode)
with open(weightfilename,"w") as file:
    for i in network.LayerList:
        for j in range(i.weights.shape[0]):
            for k in range(i.weights.shape[1]):
                file.write(str(i.weights[j][k]))
                file.write("\n")
        