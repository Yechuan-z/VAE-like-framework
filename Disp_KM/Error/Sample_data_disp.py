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
from scipy.interpolate import interp1d
from torch.nn import functional as F
import sklearn
import torch





'''
PLDS = [0.2, 0.2, 0.225, 0.3, 0.375, 0.45, 0.5, 0.55, 0.6, 0.6, 0.625, 0.625, 0.65, 0.65, 0.675,
0.675, 0.7, 0.7, 0.7, 0.7, 1.25, 1.275, 1.3, 1.35, 1.375, 1.4, 1.425, 1.425, 1.475,
1.5, 1.675, 1.75, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975] 

'''

n = 1000
rd_state = np.random.RandomState(5)
#tpts = rd_state.uniform(1.0, 5.0, size=(n,))        
delttiss = rd_state.uniform(0.25, 2, size=(n,))
tau = np.repeat(1.8,n)
t1 = np.repeat(1.3,n)
t1b = np.repeat(1.65,n)
M0_ftiss = rd_state.uniform(0.1, 2, size=(n,))*1500
log_s = np.repeat(2.0,n)
log_p = np.repeat(-2.3,n)






PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
k = len(PLDS)

'''
PLDS = [0.2, 0.2, 0.225, 0.3, 0.375, 0.45, 0.5, 0.55, 0.6, 0.6, 0.625, 0.625, 0.65, 0.65, 0.675,
0.675, 0.7, 0.7, 0.7, 0.7, 1.25, 1.275, 1.3, 1.35, 1.375, 1.4, 1.425, 1.425, 1.475,
1.5, 1.675, 1.75, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975] 

'''



TIS = np.zeros((n,k))
for j in range(k):
    for i in range(n):
        TIS[i,j] = tau[i,]+PLDS[j]



def tissue_signal(t, M0_ftiss, delt, t1, tau, t1b, log_s, log_p):
    k = 6
    conv_dt = 0.1
    conv_tmax = 5.0
    conv_nt = 1 + int(conv_tmax / conv_dt)
    conv_t = np.linspace(0.0, conv_tmax, conv_nt)
    conv_t = torch.tensor(conv_t)
    aif = aif_gammadisp(conv_t, delt, tau, t1b, log_s, log_p)
    aif= torch.flip(aif,dims=[-1])
    aif = torch.reshape(aif, (aif.shape[0],1,1,aif.shape[1],1))
    resid = resid_wellmix(conv_t, t1)
    resid = F.pad(resid,(resid.shape[1]-1,0),mode='constant')
    resid = torch.reshape(resid, (resid.shape[0],1,resid.shape[1],1))
    batch_size = aif.shape[0]
    in_channels = 1
    out_channels = 1
    o = F.conv2d( resid.view(1, batch_size*in_channels, resid.size(2), resid.size(3)).float(), aif.view(batch_size*out_channels, in_channels, aif.size(3), aif.size(4)).float(), groups=batch_size, padding = 'valid') 
    kinetic_curve = torch.squeeze(o)*conv_dt
    kc = kinetic_curve.view(kinetic_curve.size(0), 1, kinetic_curve.size(1),1)
    t_scale = 2*t/(conv_tmax) - 1
    t_scale = torch.unsqueeze(t_scale,dim = -1)
    t_scale = torch.concat((torch.zeros_like(t_scale),t_scale),-1)
    t_scale = torch.unsqueeze(t_scale,dim = 2)
    signal = F.grid_sample(kc.float(), t_scale.float(), mode='bilinear', padding_mode='zeros', align_corners=True)
    signal = torch.squeeze(signal)
    return M0_ftiss*signal



def aif_gammadisp(t, delt, tau, t1b, log_s, log_p):
    s = torch.exp(log_s)
    p = torch.exp(log_p)
    sp = s*p
    #sp = tf.clip_by_value(sp, -1e12, 10)
    pre_bolus = torch.less(t, delt)
    post_bolus = torch.greater(t, torch.add(delt, tau))
    during_bolus = torch.logical_and(torch.logical_not(pre_bolus), torch.logical_not(post_bolus))
    kcblood_nondisp = 2 * torch.exp(-delt / t1b)
    k = 1 + sp
    gamma1 = torch.igammac(k, s * (t - delt))
    gamma2 = torch.igammac(k, s * (t - delt - tau))
    kcblood = torch.zeros_like(during_bolus)
    kcblood = torch.where(during_bolus, (kcblood_nondisp * (1 - gamma1)).float(), kcblood.float())
    kcblood = torch.where(post_bolus, (kcblood_nondisp * (gamma2 - gamma1)).float(), kcblood.float())
    return kcblood


def resid_wellmix(t, t1):
    t1_app = 1 / (1 / t1 + 0.01 / 0.9)
    resid = torch.exp(-t / t1_app)
    return resid




M0_ftiss2 = torch.unsqueeze(torch.tensor(M0_ftiss),-1)
delt2 = torch.unsqueeze(torch.tensor(delttiss),-1)
t12 = torch.unsqueeze(torch.tensor(t1),-1)
t1b2 = torch.unsqueeze(torch.tensor(t1b),-1)
tau2 = torch.unsqueeze(torch.tensor(tau),-1)
log_s2 = torch.unsqueeze(torch.tensor(log_s),-1)
log_p2 = torch.unsqueeze(torch.tensor(log_p),-1)
t2 = torch.tile(torch.from_numpy(np.array(PLDS).reshape((1,k))),(n,1)) + tau2



signal = tissue_signal(t2, M0_ftiss2, delt2, t12, tau2, t1b2, log_s2, log_p2)
signal = signal.detach().numpy()



idx = np.where(signal.sum(1)!=0)[0].tolist()
n = len(idx)
TIS = TIS[idx]
delttiss = delttiss[idx]
tau = tau[idx]
t1 = t1[idx]
t1b = t1b[idx]
M0_ftiss = M0_ftiss[idx]
#snr = snr[idx]
signal = signal[idx]





df_M0_ftiss = pd.DataFrame(M0_ftiss)
df_delttiss = pd.DataFrame(delttiss)
df_tau = pd.DataFrame(tau)
df_t1 = pd.DataFrame(t1)
df_t1b = pd.DataFrame(t1b)
df_signal = pd.DataFrame(signal)
df_log_s = pd.DataFrame(log_s)
df_log_p = pd.DataFrame(log_p)
df_M0_ftiss.to_csv('M0_ftiss.csv')
df_delttiss.to_csv('delttiss.csv')
df_tau.to_csv('tau.csv')
df_t1b.to_csv('t1b.csv')
df_t1.to_csv('t1.csv')
df_signal.to_csv('signal.csv')
df_log_s.to_csv('log_s.csv')
df_log_p.to_csv('log_p.csv')


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





#import matplotlib.pyplot as plt
#subset = df_signal_noise.T
#subset = subset.iloc[:, :10]
#print(subset)
#subset.plot()
#plt.show()

