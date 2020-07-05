import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils import data as data_torch
import torch
class SimpleModelDep(nn.Module):
    '''
    A really simple network arcchitecture
    '''
    def __init__(self,input_size, output_size, n_layers,  layer_size, skip_connection=False):
        super(SimpleModel,self).__init__()
        
        self.input_lay=nn.Linear(input_size,layer_size)
        self.intermediate_layers:[nn.Sequential]=[]
        for n in range(n_layers):
            self.intermediate_layers.append(nn.Sequential((nn.Linear(layer_size,layer_size)),nn.ELU()))
        
        self.output_lay=nn.Linear(layer_size,output_size)
    
    def forward(self,x):
   
        x=self.input_lay(x)
        x=F.elu(x)
        for n in self.intermediate_layers:
            x=n(x)
        x=self.output_lay(x)
        return x
    

class SimpleModel(nn.Module):  
    '''
    A really simple network arcchitecture
    '''
    def __init__(self,input_size, output_size, n_layers:[],loss_func, lr=.001):
        super(SimpleModel,self).__init__()
        self.lr=lr
        self.loss_func=loss_func
        if len(n_layers)>0:
            self.input_lay=nn.Linear(input_size,n_layers[0])
            self.intermediate_layers:[nn.Sequential]=[]
            for n in range(len(n_layers)-1):
                self.intermediate_layers.append(nn.Sequential((nn.Linear(n_layers[n],n_layers[n+1])),nn.ELU()))

            self.output_lay=nn.Linear(n_layers[len(n_layers)-1],output_size)
        else:
            self.input_lay=nn.Linear(input_size,output_size)
            self.intermediate_layers=[]
            self.output_lay=None
    
    
    def setActivationFunction(): #if we want to override this in some way
        pass 
    
    def setLossFunction(self, loss_func):
        self.loss_func=loss_func 
        

    def setLearnRate(self,lr):
        self.lr=lr
    def forward(self,x):
   
        x=self.input_lay(x)
        #If I want to just linear regression effectively
        if(type(self.output_lay)!=type(None)): 
            x=F.elu(x)
    
        for n in self.intermediate_layers:
            x=n(x)
        
        if(type(self.output_lay)!=type(None)):
            x=self.output_lay(x)
           
        return (torch.tanh(x)+1.0)/2.0 # Yes i could be using SIGMOID. BBUT BAH 
    
    
            
    def _prepareLoss(self, batched_data:[]):
        '''
        This is just so i can customize the loss function and such 
        '''
        expected_output=batched_data[0][1].float()  #ground truth
        
        output=self(batched_data[0][0].float())
 
        return self.loss_func(output,expected_output)
            
    
    def train_model(self,epochs,dataloader,optimizer,is_training=True)->[float]:
        loss_vals=[]
        
        
        for epoch in range(epochs):
            cycle=(iter(dataloader))
            temp_losses_batch = []
            
            for i in range(len(cycle)):
                relevant_data=next(cycle)

                loss=self._prepareLoss([relevant_data]) #array so it can. be arbitrary features
                temp_losses_batch.append(loss.detach().numpy())
                optimizer.zero_grad()

                if is_training==True:
                    loss.backward()
                    optimizer.step()
         
            loss_vals.append(np.array(temp_losses_batch).sum())
        
        return loss_vals
        

    
    @staticmethod
    def paramCount(inputLength,outputLength, layerSizes=[50,25,25], trainingDataSize=20000):
            '''
            This is Not working, but i've. left the code for trying to usee an equation solver 
            to estimate the exact number of parameters assuming you wanted to distribute the. parameters 
            between your layers 
            Example bebllow has 3 Hidden Layers, w iht second layer being. half the first layer, and the last layer. being same as first 
            
              from sympy.solvers import solve
            from sympy import *
            x,y,z = sym.symbols('x,y,z',positive=True)
            #Assume first. layer is 200 features, and output is 100 features
            #Assume number of parameters you want to calcualte is 20000
            f = sym.Eq((200*x)+(x*y)+(y*z)+(z*100),20000)
            t= sym.Eq(y,x*.5)
            l= sym.Eq(z,x)
            solution=sym.solve([f,t,l],(x,y,z),positive=True)
            solution
            
            '''
            return None
        
    