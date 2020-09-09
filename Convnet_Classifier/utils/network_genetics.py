import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from utils.convnet_functions import *


class fc_layer_params(nn.Module):
        
    def __init__(self):
        super().__init__()

        self.fc_params = self.get_fc_params()
            
    ##################################################
    ########Functions to generate Parameters##########
    def get_fc_params(self):
        activation = random.choice([True,False])
        dropout = random.choice([True, False])
        
        params = { 'activation' : activation, 'dropout' : dropout}
        
        if activation:
            params['activ_params'] = self.get_activation_params()
        if dropout:
            params['dropout_frac'] = random.randint(2,5)/10
        return params 

        
    def get_activation_params(self):
        return random.choice([nn.ReLU(), nn.LeakyReLU()]) 
        
        
class conv_layer_params(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    
    ##################################################
    ########Functions to generate Parameters##########
    def get_conv_params(self):

        kernel =  np.array([ random.randint(2,5), random.randint(2,5), random.randint(2,5) ])
        stride = np.array([ random.randint(1,3), random.randint(1,3), random.randint(1,3) ])
        padding = np.array([ random.randint(0,1), random.randint(0,1), random.randint(0,1) ])
        bnorm = random.choice([True,False])
        pool = random.choice([True,False])
        activation1 = random.choice([True,False])
        activation2 = random.choice([True,False])
        dropout = random.choice([True, False])

        params = { 'kernel' : kernel, 'stride' : stride, 'padding' : padding, 'bnorm' : bnorm, 'pool' : pool, 
                  'activation1' : activation1, 
                  'activation2' : activation2, 
                  'dropout' : dropout}

        if pool:
            params['pool_params'] = self.get_pool_params()
        if activation1:
            params['activ1_params'] = self.get_activation_params()
        if activation2:
            params['activ2_params'] = self.get_activation_params()
        if dropout:
            params['dropout_frac'] = random.randint(2,5)/10
        return params 

    
    def get_pool_params(self, kernel = None, stride = None):
        if not kernel:
            kernel =  np.array([ random.randint(2,5), random.randint(2,5), random.randint(2,5) ])
        if not stride:
            stride = np.array([ random.randint(1,3), random.randint(1,3), random.randint(1,3) ])

        params = { 'kernel' : kernel, 'stride' : stride }
        return params
    
    def get_activation_params(self):
        return random.choice([nn.ReLU(), nn.LeakyReLU()])   
    

class gene_aggregator(conv_layer_params, fc_layer_params):
    def __init__(self):
        super().__init__()
        
        self.num_convs = self.get_num_convs(num=None)
        self.num_fc = self.get_num_fc(num=None)
        
        #Random init
        self.convs = []
        for i in range(self.num_convs):
            self.convs.append(self.get_conv_params())
        self.fcs = []
        for i in range(self.num_fc):
            self.fcs.append(self.get_fc_params())
            
        self.get_conv_channels(num=None)
        
        self.get_conv_shapes()
        self.get_fc_features()

    def get_conv_channels(self,num):
        if not num:
            num = random.randint(2,5)

        for i in range(self.num_convs):
            self.convs[i]['channels_out'] = 2**(num+i)
            if i==0:
                self.convs[i]['channels_in'] = 1
            else:
                self.convs[i]['channels_in'] = self.convs[i-1]['channels_out']        
                
    def get_fc_features(self):
        features_in = int( np.prod(self.convs[self.num_convs-1]['shape_out'])*self.convs[self.num_convs-1]['channels_out'] )
        divisor = features_in**(1/self.num_fc)
        
        for i in range(self.num_fc):
            if i==0:
                self.fcs[i]['features_in'] = features_in
            else:
                self.fcs[i]['features_in'] = self.fcs[i-1]['features_out']
            self.fcs[i]['features_out'] = math.ceil(self.fcs[i]['features_in'] / divisor)
            
        #Make sure last channel out is 1
        self.fcs[self.num_fc-1]['features_out']=1
                    
    def get_num_convs(self, num):
        if num:
            return num
        else:
            return random.randint(1,5)
        
    def get_num_fc(self, num):
        if num:
            return num
        else:
            return random.randint(1,5)
        
    def get_conv_layer_shape_out(self, layer, shape_in = np.array([106,106,120]) ):
        
        convshape = np.ceil( (shape_in + 2*layer['padding'] - layer['kernel']) / layer['stride'])
        
        if layer['pool']:
            poolshape = np.ceil( (convshape - layer['pool_params']['kernel']) / layer['pool_params']['stride'] )
            return poolshape
        else:
            return convshape
        
 
    def get_conv_shapes(self, shape_in = np.array([106,106,120])):
        
        for i in range(self.num_convs):
            if i==0:
                self.convs[i]['shape_in'] = shape_in
                self.convs[i]['shape_out'] = self.get_conv_layer_shape_out(self.convs[i])
            else:
                self.convs[i]['shape_in'] = self.convs[i-1]['shape_out']
                self.convs[i]['shape_out'] = self.get_conv_layer_shape_out(self.convs[i])
        
class network_constructor(gene_aggregator, nn.Module):
    
    def __init__(self):
        super().__init__()
        
    
    def build(self):
        
        self.built_convs=[]
        for i in range(self.num_convs):
            self.built_convs.append(self.make_conv_layer(self.convs[i]))
        
        self.built_fcs=[]
        for i in range(self.num_fc):
            self.built_fcs.append(self.make_fc_layer(self.fcs[i]))
    
    def forward(self,x):
        
        #Run Convs
        for i in range(self.num_convs):
            x = self.built_convs[i](x)
        
        #Flatten and get size
        x = torch.flatten(x, start_dim=1)
        xsh = x.size(-1)
        x = x.view(-1,xsh)
        
        #Run FC
        for i in range(self.num_fc):
            x = self.built_fcs[i](x)
        return x
    
    def make_conv_layer(self, params):
        conv = nn.Sequential()

        conv.add_module('conv', nn.Conv3d(params['channels_in'], params['channels_out'], params['kernel'], params['stride'], params['padding']) ) 
        if params['activation1']:
            conv.add_module('activation1', params['activ1_params'])
        if params['bnorm']:
            conv.add_module('BatchNorm', nn.BatchNorm3d(params['channels_out']) )
        if params['activation2']:
            conv.add_module('activation2', params['activ2_params'])
        if params['pool']:
            #I do not know why stride has to be a tuple
            conv.add_module('pool', nn.MaxPool3d(params['pool_params']['kernel'], stride=tuple(params['pool_params']['stride'])) )
        if params['dropout']:
            conv.add_module('dropout', nn.Dropout(params['dropout_frac']) )
        return conv
    
    def make_fc_layer(self, params):
        fc = nn.Sequential()
        
        fc.add_module('linear', nn.Linear(params['features_in'], params['features_out']) )
        if params['activation']:
            fc.add_module('activation', params['activ_params'])
        if params['dropout']:
            fc.add_module('dropout', nn.Dropout(params['dropout_frac']) )
        return fc
    