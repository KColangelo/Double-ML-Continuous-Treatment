# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:04:09 2020
Last update Monday Jan 10 12:11 pm 2022

This file defines the neural network models used for both the simulations and
empirical application in Colangelo and Lee (2021). PyTorch is the package used
for all neural networks. We begin by defining a class "NeuralNet" which creates
a torch neural network that adheres to the requirements of the DDMLCT class. 
That is, DDMLCT requires models which have both a .fit and .predict method,
so NeuralNet creates this for us. We also define class NeuralNetk as the base
class for the K neural network which utilizes the additional kernel function
in the loss function as specified in Colangelo and Lee (2021).
We then define a subclass of NeuralNet for each of the 4 models used in the 
simulations and empirical application. To define a new neural network model 
which can be used with the DDMLCT class, you need only define a subclass of 
NeuralNet or NeuralNetk and define the layers as has been done below, and 
specify the criterion, optimizer, and number of epochs. 

"""

import torch
import torch.nn as nn
torch.set_default_tensor_type('torch.DoubleTensor')

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(20,1))
        self.layer_1.add_module("R1", nn.SELU())
        
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.5,weight_decay=0.2)
        self.epochs = 300

    def weight_reset(self,m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()   
            
    def forward(self, x):
        for i in range(1,len(vars(self)['_modules'])):
            x = getattr(self,('layer_'+str(i)))(x)

        return x
    
    def fit(self,x,y):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y),1)

        for t in range(self.epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(x)
        
            # Compute and print loss
            loss = self.criterion(y_pred, y)
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self
            
    def predict(self,x):
        x = torch.from_numpy(x)
        pred = self(x).detach().numpy()
        return pred[:,0]

class NeuralNetk(torch.nn.Module):
    def __init__(self):
        super(NeuralNetk, self).__init__()
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(20,1))
        self.layer_1.add_module("R1", nn.SELU())
        
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.5,weight_decay=0.2)
        self.epochs = 300

    def weight_reset(self,m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()   
            
    def forward(self, x):
        for i in range(1,len(vars(self)['_modules'])):
            x = getattr(self,('layer_'+str(i)))(x)


        return x
    def my_loss(self,output, target,K):
        loss = torch.mean(((1/2)*(output - target)**2)*K)
        return loss
    def fit(self,x,y,K):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y),1)
        K = torch.from_numpy(K)
        for t in range(self.epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(x)
        
            # Compute and print loss
            loss = self.my_loss(y_pred,y,K)
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self
    def predict(self,x):
        x = torch.from_numpy(x)
        pred = self(x).detach().numpy()
        return pred[:,0]
    
class NeuralNet1(NeuralNet):
    def __init__(self,k):
        super(NeuralNet1, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("L2", nn.Linear(10,10))
        self.layer_2.add_module("R2", nn.ReLU())
        
        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("L3", nn.Linear(10,10))
        self.layer_3.add_module("R3", nn.ReLU())
        
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("L4", nn.Linear(10,10))
        self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_5 = torch.nn.Linear(10,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.5,weight_decay=0.2)

class NeuralNet1k(NeuralNetk):
    def __init__(self,k):
        super(NeuralNet1k, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("L2", nn.Linear(10,10))
        self.layer_2.add_module("R2", nn.ReLU())
        
        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("L3", nn.Linear(10,10))
        self.layer_3.add_module("R3", nn.ReLU())
        
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("L4", nn.Linear(10,10))
        self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_5 = torch.nn.Linear(10,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.5,weight_decay=0.2)
 
class NeuralNet2(NeuralNet):
    def __init__(self,k):
        super(NeuralNet2, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("L2", nn.Linear(10,10))
        self.layer_2.add_module("R2", nn.ReLU())
        
        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("L3", nn.Linear(10,10))
        self.layer_3.add_module("R3", nn.ReLU())
        
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("L4", nn.Linear(10,10))
        self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_5 = torch.nn.Linear(10,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.5,weight_decay=0.2)

      
class NeuralNet3(NeuralNet):
    def __init__(self,k):

        super(NeuralNet3, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,100))
        self.layer_1.add_module("R1", nn.SELU())
        
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("L2", nn.Linear(100,20))
        self.layer_2.add_module("R2", nn.SELU())
        
        self.layer_3= torch.nn.Linear(20,1)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.0,weight_decay=0.1)
        #self.epochs = 100  
        
        
class NeuralNet4(NeuralNet):
    def __init__(self,k):

        super(NeuralNet4, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("L2", nn.Linear(10,10))
        self.layer_2.add_module("R2", nn.ReLU())
        
        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("L3", nn.Linear(10,10))
        self.layer_3.add_module("R3", nn.ReLU())
        
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("L4", nn.Linear(10,10))
        self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_5 = torch.nn.Linear(10,1)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.0,weight_decay=0.1)

        
class NeuralNet3k(NeuralNet):
    def __init__(self,k):

        super(NeuralNet3k, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,20))
        self.layer_1.add_module("R1", nn.ReLU())
        
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("L2", nn.Linear(20,20))
        self.layer_2.add_module("R2", nn.ReLU())
        
        self.layer_3= torch.nn.Linear(20,1)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01,momentum=0.0,weight_decay=0.1)
        #self.epochs = 100  
    def my_loss(self,output, target,K):
        loss = torch.mean(((1/2)*(output - target)**2)*K)
        return loss
    def fit(self,x,y,K):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y),1)
        K = torch.from_numpy(K)
        for t in range(self.epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(x)
        
            # Compute and print loss
            loss = self.my_loss(y_pred,y,K)
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self
         