import os.path
import subprocess
import numpy as np
import sklearn.metrics
import pandas as pd
import nibabel as nib
import logging
logging.getLogger().setLevel(logging.INFO)
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.metrics import r2_score
import sklearn
import torch


'''
PLDS = [0.2, 0.2, 0.225, 0.3, 0.375, 0.45, 0.5, 0.55, 0.6, 0.6, 0.625, 0.625, 0.65, 0.65, 0.675,
0.675, 0.7, 0.7, 0.7, 0.7, 1.25, 1.275, 1.3, 1.35, 1.375, 1.4, 1.425, 1.425, 1.475,
1.5, 1.675, 1.75, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975] 

'''



n = 1000
rd_state = np.random.RandomState(5)        
delttiss = rd_state.uniform(0, 2, size=(n,))
M0_ftiss  = rd_state.uniform(0, 2, size=(n,))*1500
tau = np.repeat(1.8,n)
t1 = np.repeat(1.3,n)
t1b = np.repeat(1.65,n)


PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
k = len(PLDS)



TIS = np.zeros((n,k))
for j in range(k):
    for i in range(n):
        TIS[i,j] = tau[i,]+PLDS[j]


def tissue_signal(t, M0_ftiss, delt, t1, t1b, tau):
    post_bolus = torch.greater(t, torch.add(tau, delt))
    during_bolus = torch.logical_and(torch.greater(t, delt), torch.logical_not(post_bolus))
    t1_app = 1 / (1 / t1 + 0.01 / 0.9)
    factor = 2 * t1_app * torch.exp(-delt / t1b)
    during_bolus_signal =  factor * (1 - torch.exp(-(t - delt) / t1_app))
    post_bolus_signal = factor * torch.exp(-(t - tau - delt) / t1_app) * (1 - torch.exp(-tau / t1_app))
    signal = torch.zeros(during_bolus_signal.shape)
    signal = torch.where(during_bolus, during_bolus_signal, signal.double())
    signal = torch.where(post_bolus, post_bolus_signal, signal.double())
    out = M0_ftiss*signal
    return out



ftiss2 = torch.tile(torch.unsqueeze(torch.tensor(M0_ftiss),-1),(1,k))
delttiss2 = torch.tile(torch.unsqueeze(torch.tensor(delttiss),-1),(1,k))
t12 = torch.tile(torch.unsqueeze(torch.tensor(t1),-1),(1,k))
t1b2 = torch.tile(torch.unsqueeze(torch.tensor(t1b),-1),(1,k))
tau2 = torch.tile(torch.unsqueeze(torch.tensor(tau),-1),(1,k))
t2 = torch.tile(torch.from_numpy(np.array(PLDS).reshape((1,k))),(n,1)) + tau2

signal = tissue_signal(t2,ftiss2, delttiss2, t12, t1b2, tau2)
signal = signal.detach().numpy()

idx = np.where(signal.sum(1)!=0)[0].tolist()
n = len(idx)
TIS = TIS[idx]
delttiss = delttiss[idx]
tau = tau[idx]
t1 = t1[idx]
t1b = t1b[idx]
M0_ftiss = M0_ftiss[idx]
signal = signal[idx]

df_M0_ftiss = pd.DataFrame(M0_ftiss)
df_delttiss = pd.DataFrame(delttiss)
df_tau = pd.DataFrame(tau)
df_t1 = pd.DataFrame(t1)
df_t1b = pd.DataFrame(t1b)
df_signal = pd.DataFrame(signal)
df_M0_ftiss.to_csv('M0_ftiss.csv')
df_delttiss.to_csv('delttiss.csv')
df_tau.to_csv('tau.csv')
df_t1b.to_csv('t1b.csv')
df_t1.to_csv('t1.csv')
df_signal.to_csv('signal.csv')

for snr in (1000000, 10, 5, 2.5):
    e_std = np.repeat(0.6/snr,n)*1500
    e_log = np.log(e_std)
    print("Noise STD: ", e_std)
    signal_noise = np.random.normal(signal, np.tile(np.reshape(e_std,(n,1)),(1,len(PLDS))))
    e_log = e_log[idx]
    e_std = e_std[idx]
    df_e_log = pd.DataFrame(e_log)
    df_e_std = pd.DataFrame(e_std)
    df_signal_noise = pd.DataFrame(signal_noise)
    df_e_log.to_csv('e_log.csv')
    df_e_std.to_csv('e_std.csv')
    df_signal_noise.to_csv('signal_noise_snr_%i.csv' % (snr*10))

