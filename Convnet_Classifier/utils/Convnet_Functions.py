#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def Calc_Dims_3d(shape, padding, kernel, stride, Pool_kernel, Pool_stride):
    convshape = np.ceil((shape + 2*padding - kernel) / stride)
    poolshape = np.ceil((convshape - Pool_kernel) / Pool_stride)
    
    return poolshape

