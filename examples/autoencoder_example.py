# os stuff
import os
import sys
sys.path.append('..')

# whobpyt stuff
import whobpyt
from whobpyt.data.dataload import dataloader
from whobpyt.models.RWW.wong_wang import ParamsRWW
from whobpyt.models.RWW.wong_wang import RNNRWW
from whobpyt.optimization.modelfitting import Model_fitting
from whobpyt.optimization.custom_cost_RWW import CostsRWW

# array and pd stuff
import numpy as np
import numpy.ma as ma
import pandas as pd
import pickle


ts_dir = '/Users/ClemensP/Desktop/FOR_data/' 
sc_dir = '/Users/ClemensP/Desktop/FOR_data/'
sub = '0161'
sc_file = sc_dir + sub + '_SC_weights.txt'
ts_file = ts_dir + sub + '_timeseries_RS.txt'
sc = np.loadtxt(sc_file) #np function
SC = (sc + sc.T) * 0.5
sc = np.log1p(SC) / np.linalg.norm(np.log1p(SC)) #np function
sc = sc[14:,14:]
#sc = torch.as_tensor(sc)
ts = np.loadtxt(ts_file) #np function
ts = ts / np.max(ts) #np function
ts = ts[14:]
emp = ts.T
fc_emp = np.corrcoef(ts) #np function

# define options for wong-wang model
node_size = fc_emp.shape[0]
#mask = np.tril_indices(node_size, -1) #np function
num_epoches = 1
batch_size = int(ts.shape[1]/13)
step_size = 0.05
input_size = 2
tr = 2
repeat_size = 5
model_name = 'RWW'

data_mean = dataloader(ts.T, num_epoches, batch_size)
par = ParamsRWW()
model = RNNRWW(node_size, batch_size, step_size, repeat_size, tr, sc, False, par)
ObjFun = CostsRWW()
F = Model_fitting(model, data_mean, num_epoches, ObjFun)
model.setModelParameters()
F.test(20)
F.save('/Users/ClemensP/Desktop/FOR_data/output/'+sub+'_output')