from termios import FF1
import numpy as np
import pandas as pd
import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import r2_score
import os.path


import sys
import math
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import nibabel as nib
from sklearn.metrics import r2_score
import sklearn
from ssl import Options
import sys
import scipy.stats as ss





############################################# VAE

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int, #number of neurons in input layer
        planes: int, #number of neurons in hidden layer
    ) -> None:
        super().__init__() != 1
        self.Linear1 = nn.Linear(planes, planes)
        #self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.Linear2 = nn.Linear(planes, planes)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.Linear1(x)
        out = self.relu(out)
        out = self.Linear2(out)
        out += identity
        out = self.relu(out)
        return out







class VAE(pl.LightningModule):
    def __init__(self, input_dim= 9, enc_out_dim=100, latent_dim=5):
        super().__init__()
        self.save_hyperparameters()
        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim , enc_out_dim),
            nn.ReLU(),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
            BasicBlock(enc_out_dim,enc_out_dim),
        )
        self.decoder = self.tissue_signal
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_std = nn.Linear(enc_out_dim, latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0]))
    def tissue_signal(self, t, M0_ftiss, delt, t1, tau, t1b, log_s, log_p):
        conv_dt =torch.tensor(0.1) 
        conv_tmax = torch.tensor(5.0)
        #conv_nt = 1 + int(conv_tmax / conv_dt)
        conv_nt = 51
        conv_t = torch.linspace(0.0, conv_tmax, conv_nt)
        #conv_t = torch.tensor(conv_t)
        sampling_size = M0_ftiss.shape[0]
        sample_size = M0_ftiss.shape[1]
        k = t.shape[-1]
        aif = self.aif_gammadisp(conv_t, delt, tau, t1b, log_s, log_p)
        aif = torch.reshape(aif, (sampling_size*sample_size,aif.shape[-1]))
        aif= torch.flip(aif,dims=[-1])
        aif = torch.reshape(aif, (aif.shape[0],1,1,aif.shape[1],1))
        #aif = torch.zeros_like(aif)
        resid = self.resid_wellmix(conv_t, t1)
        #print('resid',resid.mean())
        resid = torch.reshape(resid, (sampling_size*sample_size,resid.shape[-1]))
        resid = F.pad(resid,(resid.shape[1]-1,0),mode='constant')
        resid = torch.reshape(resid, (resid.shape[0],1,resid.shape[1],1))
        batch_size = aif.shape[0]
        print('aif',aif.mean())
        print('resid',resid.mean())
        in_channels = 1
        out_channels = 1
        o = F.conv2d( resid.view(1, batch_size*in_channels, resid.size(2), resid.size(3)).float(), aif.view(batch_size*out_channels, in_channels, aif.size(3), aif.size(4)).float(), groups=batch_size, padding = 'valid') 
        kinetic_curve = torch.squeeze(o)*conv_dt
        kc = kinetic_curve.view(kinetic_curve.size(0), 1, kinetic_curve.size(1),1)
        #kc = nn.ReLU()(kc)
        #kc = 10 - nn.ReLU()(5-kc)
        t = torch.reshape(t, (sampling_size*sample_size,t.shape[-1]))
        t_scale = 2*t/(conv_tmax) - 1
        t_scale = torch.unsqueeze(t_scale,dim = -1)
        t_scale = torch.concat((torch.zeros_like(t_scale),t_scale),-1)
        t_scale = torch.unsqueeze(t_scale,dim = 2)
        signal = F.grid_sample(kc.float(), t_scale.float(), mode='bilinear', padding_mode='zeros', align_corners=True)
        signal = torch.squeeze(signal)
        signal = torch.reshape(signal, (sampling_size, sample_size,k))
        print('signal',signal.mean())
        return M0_ftiss*signal
    def aif_gammadisp(self, t, delt, tau, t1b, log_s, log_p):
        s = torch.exp(torch.clip(log_s,max = 10))
        p = torch.exp(torch.clip(log_p,max = 10))
        sp = s*p
        pre_bolus = torch.less(t, delt)
        post_bolus = torch.greater(t, torch.add(delt, tau))
        during_bolus = torch.logical_and(torch.logical_not(pre_bolus), torch.logical_not(post_bolus))
        kcblood_nondisp = 2 * torch.exp(-delt / t1b)
        k = 1 + sp
        gamma1 = torch.igammac(nn.ReLU()(k.detach()), nn.ReLU()(s * (t - delt)))
        gamma2 = torch.igammac(nn.ReLU()(k.detach()), nn.ReLU()(s * (t - delt - tau)))
        print('k',k.mean())
        print('(s * (t - delt))',(s * (t - delt)).mean())
        print('gamma1',gamma1.mean())
        print('gamma2',gamma2.mean())
        kcblood = torch.zeros_like(during_bolus)
        kcblood = torch.where(during_bolus, (kcblood_nondisp * (1 - gamma1)).float(), kcblood.float())
        kcblood = torch.where(post_bolus, (kcblood_nondisp * (gamma2 - gamma1)).float(), kcblood.float())
        return kcblood
    def resid_wellmix(self, t, t1):
        t1_app = 1 / (1 / t1 + 0.01 / 0.9)
        resid = torch.exp(-t / t1_app)
        return resid





########################## Load VAE
path = 'General_KM/VAE_like_general.pth'
vae = torch.load(path)



########################## Load Data
n = 64*64*24

data_0 = np.reshape(np.array(nib.load('asldata_diff_0.nii.gz').dataobj),(n,6))
data_1 = np.reshape(np.array(nib.load('asldata_diff_1.nii.gz').dataobj),(n,6))
data_2 = np.reshape(np.array(nib.load('asldata_diff_2.nii.gz').dataobj),(n,6))
data_3 = np.reshape(np.array(nib.load('asldata_diff_3.nii.gz').dataobj),(n,6))
data_4 = np.reshape(np.array(nib.load('asldata_diff_4.nii.gz').dataobj),(n,6))
data_5 = np.reshape(np.array(nib.load('asldata_diff_5.nii.gz').dataobj),(n,6))
data_6 = np.reshape(np.array(nib.load('asldata_diff_6.nii.gz').dataobj),(n,6))
data_7 = np.reshape(np.array(nib.load('asldata_diff_7.nii.gz').dataobj),(n,6))
mask = np.array(nib.load('mask.nii.gz').dataobj)
tau = np.repeat(1.4,n)
t1 = np.repeat(1.3,n)
t1b = np.repeat(1.6,n)



x_data_0 = np.zeros((n,8+data_0.shape[1]),np.float32)
sig_dim = data_0.shape[1]
x_data_0[:,0:sig_dim] = np.reshape(data_0,(n,data_0.shape[1]))
x_data_0[:,sig_dim] = np.reshape(tau,(n,))
x_data_0[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_0[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_0[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_0 = torch.from_numpy(x_data_0) 


x_data_1 = np.zeros((n,8+data_1.shape[1]),np.float32)
sig_dim = data_1.shape[1]
x_data_1[:,0:sig_dim] = np.reshape(data_1,(n,data_1.shape[1]))
x_data_1[:,sig_dim] = np.reshape(tau,(n,))
x_data_1[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_1[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_1[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_1 = torch.from_numpy(x_data_1) 


x_data_2 = np.zeros((n,8+data_2.shape[1]),np.float32)
sig_dim = data_2.shape[1]
x_data_2[:,0:sig_dim] = np.reshape(data_2,(n,data_2.shape[1]))
x_data_2[:,sig_dim] = np.reshape(tau,(n,))
x_data_2[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_2[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_2[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_2 = torch.from_numpy(x_data_2) 


x_data_3 = np.zeros((n,8+data_3.shape[1]),np.float32)
sig_dim = data_3.shape[1]
x_data_3[:,0:sig_dim] = np.reshape(data_3,(n,data_3.shape[1]))
x_data_3[:,sig_dim] = np.reshape(tau,(n,))
x_data_3[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_3[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_3[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_3 = torch.from_numpy(x_data_3) 


x_data_4 = np.zeros((n,8+data_4.shape[1]),np.float32)
sig_dim = data_4.shape[1]
x_data_4[:,0:sig_dim] = np.reshape(data_4,(n,data_4.shape[1]))
x_data_4[:,sig_dim] = np.reshape(tau,(n,))
x_data_4[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_4[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_4[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_4 = torch.from_numpy(x_data_4) 


x_data_4 = np.zeros((n,8+data_4.shape[1]),np.float32)
sig_dim = data_4.shape[1]
x_data_4[:,0:sig_dim] = np.reshape(data_4,(n,data_4.shape[1]))
x_data_4[:,sig_dim] = np.reshape(tau,(n,))
x_data_4[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_4[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_4[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_4 = torch.from_numpy(x_data_4) 


x_data_5 = np.zeros((n,8+data_5.shape[1]),np.float32)
sig_dim = data_5.shape[1]
x_data_5[:,0:sig_dim] = np.reshape(data_5,(n,data_5.shape[1]))
x_data_5[:,sig_dim] = np.reshape(tau,(n,))
x_data_5[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_5[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_5[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_5 = torch.from_numpy(x_data_5) 


x_data_6 = np.zeros((n,8+data_6.shape[1]),np.float32)
sig_dim = data_6.shape[1]
x_data_6[:,0:sig_dim] = np.reshape(data_6,(n,data_6.shape[1]))
x_data_6[:,sig_dim] = np.reshape(tau,(n,))
x_data_6[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_6[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_6[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_6 = torch.from_numpy(x_data_6) 


x_data_7 = np.zeros((n,8+data_7.shape[1]),np.float32)
sig_dim = data_7.shape[1]
x_data_7[:,0:sig_dim] = np.reshape(data_7,(n,data_7.shape[1]))
x_data_7[:,sig_dim] = np.reshape(tau,(n,))
x_data_7[:,sig_dim+1] = np.reshape(t1,(n,))
x_data_7[:,sig_dim+2] = np.reshape(t1b,(n,))
#x_data_7[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_7 = torch.from_numpy(x_data_7)






######################################################## Comparison

######################## Run VAE
idx_mask = np.where(np.reshape(mask,(n,)) == 0)
j = 10
scaling = 200/(1090*0.85)
thr = 0.06

pred_encoder_0 = vae.encoder(x_data_0[:,0:9])
mu_0 = vae.fc_mu(pred_encoder_0)
sigma_0 = torch.exp(vae.fc_log_std(pred_encoder_0))
f_mu_0 = mu_0[:,0].detach().numpy()
att_mu_0 = mu_0[:,1].detach().numpy()
e_mu_0 = mu_0[:,2].detach().numpy()
log_s_mu_0 = mu_0[:,3].detach().numpy()
log_p_mu_0 = mu_0[:,4].detach().numpy()
f_sigma_0 = sigma_0[:,0].detach().numpy()
att_sigma_0 = sigma_0[:,1].detach().numpy()
e_sigma_0 = sigma_0[:,2].detach().numpy()
log_s_sigma_0 = sigma_0[:,3].detach().numpy()
log_p_sigma_0 = sigma_0[:,4].detach().numpy()
f_mu_0[idx_mask] = 0
att_mu_0[idx_mask] = 0
e_mu_0[idx_mask] = 0
log_s_mu_0[idx_mask] = 0 
log_p_mu_0[idx_mask] = 0  
f_sigma_0[idx_mask] = 0
att_sigma_0[idx_mask] = 0
e_sigma_0[idx_mask] = 0
log_s_sigma_0[idx_mask] = 0
log_p_sigma_0[idx_mask] = 0

mean_f_vae_0 = (np.reshape(f_mu_0,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_0 = np.reshape(att_mu_0,(64,64,24))[:,:,j:j+1]
mean_e_vae_0 = (np.reshape(np.exp(e_mu_0 + e_sigma_0**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_0 = np.reshape(log_s_mu_0,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_0 = np.reshape(log_p_mu_0,(64,64,24))[:,:,j:j+1]
std_f_vae_0 = (np.reshape(f_sigma_0,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_0 = np.reshape(att_sigma_0,(64,64,24))[:,:,j:j+1]
std_e_vae_0 = (np.reshape(np.sqrt((np.exp(e_sigma_0**2)-1)*np.exp(2*e_mu_0 + e_sigma_0**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_0 = np.reshape(log_s_sigma_0,(64,64,24))[:,:,j:j+1]
std_log_p_vae_0 = np.reshape(log_p_sigma_0,(64,64,24))[:,:,j:j+1]
mean_s_vae_0 = np.exp(mean_log_s_vae_0)
mean_s_vae_0[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_0 = np.exp(mean_log_p_vae_0)
mean_p_vae_0[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_0[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_0[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_0[np.where(mean_f_vae_0 < thr)] = 0
std_att_vae_0[np.where(mean_f_vae_0 < thr)] = 0


pred_encoder_1 = vae.encoder(x_data_1[:,0:9])
mu_1 = vae.fc_mu(pred_encoder_1)
sigma_1 = torch.exp(vae.fc_log_std(pred_encoder_1))
f_mu_1 = mu_1[:,0].detach().numpy()
att_mu_1 = mu_1[:,1].detach().numpy()
e_mu_1 = mu_1[:,2].detach().numpy()
log_s_mu_1 = mu_1[:,3].detach().numpy()
log_p_mu_1 = mu_1[:,4].detach().numpy()
f_sigma_1 = sigma_1[:,0].detach().numpy()
att_sigma_1 = sigma_1[:,1].detach().numpy()
e_sigma_1 = sigma_1[:,2].detach().numpy()
log_s_sigma_1 = sigma_1[:,3].detach().numpy()
log_p_sigma_1 = sigma_1[:,4].detach().numpy()
f_mu_1[idx_mask] = 0
att_mu_1[idx_mask] = 0
e_mu_1[idx_mask] = 0
log_s_mu_1[idx_mask] = 0 
log_p_mu_1[idx_mask] = 0  
f_sigma_1[idx_mask] = 0
att_sigma_1[idx_mask] = 0
e_sigma_1[idx_mask] = 0
log_s_sigma_1[idx_mask] = 0
log_p_sigma_1[idx_mask] = 0

mean_f_vae_1 = (np.reshape(f_mu_1,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_1 = np.reshape(att_mu_1,(64,64,24))[:,:,j:j+1]
mean_e_vae_1 = (np.reshape(np.exp(e_mu_1 + e_sigma_1**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_1 = np.reshape(log_s_mu_1,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_1 = np.reshape(log_p_mu_1,(64,64,24))[:,:,j:j+1]
std_f_vae_1 = (np.reshape(f_sigma_1,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_1 = np.reshape(att_sigma_1,(64,64,24))[:,:,j:j+1]
std_e_vae_1 = (np.reshape(np.sqrt((np.exp(e_sigma_1**2)-1)*np.exp(2*e_mu_1 + e_sigma_1**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_1 = np.reshape(log_s_sigma_1,(64,64,24))[:,:,j:j+1]
std_log_p_vae_1 = np.reshape(log_p_sigma_1,(64,64,24))[:,:,j:j+1]
mean_s_vae_1 = np.exp(mean_log_s_vae_1)
mean_s_vae_1[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_1 = np.exp(mean_log_p_vae_1)
mean_p_vae_1[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_1[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_1[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_1[np.where(mean_f_vae_1 < thr)] = 0
std_att_vae_1[np.where(mean_f_vae_1 < thr)] = 0


pred_encoder_2 = vae.encoder(x_data_2[:,0:9])
mu_2 = vae.fc_mu(pred_encoder_2)
sigma_2 = torch.exp(vae.fc_log_std(pred_encoder_2))
f_mu_2 = mu_2[:,0].detach().numpy()
att_mu_2 = mu_2[:,1].detach().numpy()
e_mu_2 = mu_2[:,2].detach().numpy()
log_s_mu_2 = mu_2[:,3].detach().numpy()
log_p_mu_2 = mu_2[:,4].detach().numpy()
f_sigma_2 = sigma_2[:,0].detach().numpy()
att_sigma_2 = sigma_2[:,1].detach().numpy()
e_sigma_2 = sigma_2[:,2].detach().numpy()
log_s_sigma_2 = sigma_2[:,3].detach().numpy()
log_p_sigma_2 = sigma_2[:,4].detach().numpy()
f_mu_2[idx_mask] = 0
att_mu_2[idx_mask] = 0
e_mu_2[idx_mask] = 0
log_s_mu_2[idx_mask] = 0 
log_p_mu_2[idx_mask] = 0  
f_sigma_2[idx_mask] = 0
att_sigma_2[idx_mask] = 0
e_sigma_2[idx_mask] = 0
log_s_sigma_2[idx_mask] = 0
log_p_sigma_2[idx_mask] = 0

mean_f_vae_2 = (np.reshape(f_mu_2,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_2 = np.reshape(att_mu_2,(64,64,24))[:,:,j:j+1]
mean_e_vae_2 = (np.reshape(np.exp(e_mu_2 + e_sigma_2**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_2 = np.reshape(log_s_mu_2,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_2 = np.reshape(log_p_mu_2,(64,64,24))[:,:,j:j+1]
std_f_vae_2 = (np.reshape(f_sigma_2,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_2 = np.reshape(att_sigma_2,(64,64,24))[:,:,j:j+1]
std_e_vae_2 = (np.reshape(np.sqrt((np.exp(e_sigma_2**2)-1)*np.exp(2*e_mu_2 + e_sigma_2**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_2 = np.reshape(log_s_sigma_2,(64,64,24))[:,:,j:j+1]
std_log_p_vae_2 = np.reshape(log_p_sigma_2,(64,64,24))[:,:,j:j+1]
mean_s_vae_2 = np.exp(mean_log_s_vae_2)
mean_s_vae_2[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_2 = np.exp(mean_log_p_vae_2)
mean_p_vae_2[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_2[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_2[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_2[np.where(mean_f_vae_2 < thr)] = 0
std_att_vae_2[np.where(mean_f_vae_2 < thr)] = 0


pred_encoder_3 = vae.encoder(x_data_3[:,0:9])
mu_3 = vae.fc_mu(pred_encoder_3)
sigma_3 = torch.exp(vae.fc_log_std(pred_encoder_3))
f_mu_3 = mu_3[:,0].detach().numpy()
att_mu_3 = mu_3[:,1].detach().numpy()
e_mu_3 = mu_3[:,2].detach().numpy()
log_s_mu_3 = mu_3[:,3].detach().numpy()
log_p_mu_3 = mu_3[:,4].detach().numpy()
f_sigma_3 = sigma_3[:,0].detach().numpy()
att_sigma_3 = sigma_3[:,1].detach().numpy()
e_sigma_3 = sigma_3[:,2].detach().numpy()
log_s_sigma_3 = sigma_3[:,3].detach().numpy()
log_p_sigma_3 = sigma_3[:,4].detach().numpy()
f_mu_3[idx_mask] = 0
att_mu_3[idx_mask] = 0
e_mu_3[idx_mask] = 0
log_s_mu_3[idx_mask] = 0 
log_p_mu_3[idx_mask] = 0  
f_sigma_3[idx_mask] = 0
att_sigma_3[idx_mask] = 0
e_sigma_3[idx_mask] = 0
log_s_sigma_3[idx_mask] = 0
log_p_sigma_3[idx_mask] = 0

mean_f_vae_3 = (np.reshape(f_mu_3,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_3 = np.reshape(att_mu_3,(64,64,24))[:,:,j:j+1]
mean_e_vae_3 = (np.reshape(np.exp(e_mu_3 + e_sigma_3**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_3 = np.reshape(log_s_mu_3,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_3 = np.reshape(log_p_mu_3,(64,64,24))[:,:,j:j+1]
std_f_vae_3 = (np.reshape(f_sigma_3,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_3 = np.reshape(att_sigma_3,(64,64,24))[:,:,j:j+1]
std_e_vae_3 = (np.reshape(np.sqrt((np.exp(e_sigma_3**2)-1)*np.exp(2*e_mu_3 + e_sigma_3**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_3 = np.reshape(log_s_sigma_3,(64,64,24))[:,:,j:j+1]
std_log_p_vae_3 = np.reshape(log_p_sigma_3,(64,64,24))[:,:,j:j+1]
mean_s_vae_3 = np.exp(mean_log_s_vae_3)
mean_s_vae_3[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_3 = np.exp(mean_log_p_vae_3)
mean_p_vae_3[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_3[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_3[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_3[np.where(mean_f_vae_3 < thr)] = 0
std_att_vae_3[np.where(mean_f_vae_3 < thr)] = 0


pred_encoder_4 = vae.encoder(x_data_4[:,0:9])
mu_4 = vae.fc_mu(pred_encoder_4)
sigma_4 = torch.exp(vae.fc_log_std(pred_encoder_4))
f_mu_4 = mu_4[:,0].detach().numpy()
att_mu_4 = mu_4[:,1].detach().numpy()
e_mu_4 = mu_4[:,2].detach().numpy()
log_s_mu_4 = mu_4[:,3].detach().numpy()
log_p_mu_4 = mu_4[:,4].detach().numpy()
f_sigma_4 = sigma_4[:,0].detach().numpy()
att_sigma_4 = sigma_4[:,1].detach().numpy()
e_sigma_4 = sigma_4[:,2].detach().numpy()
log_s_sigma_4 = sigma_4[:,3].detach().numpy()
log_p_sigma_4 = sigma_4[:,4].detach().numpy()
f_mu_4[idx_mask] = 0
att_mu_4[idx_mask] = 0
e_mu_4[idx_mask] = 0
log_s_mu_4[idx_mask] = 0 
log_p_mu_4[idx_mask] = 0  
f_sigma_4[idx_mask] = 0
att_sigma_4[idx_mask] = 0
e_sigma_4[idx_mask] = 0
log_s_sigma_4[idx_mask] = 0
log_p_sigma_4[idx_mask] = 0

mean_f_vae_4 = (np.reshape(f_mu_4,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_4 = np.reshape(att_mu_4,(64,64,24))[:,:,j:j+1]
mean_e_vae_4 = (np.reshape(np.exp(e_mu_4 + e_sigma_4**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_4 = np.reshape(log_s_mu_4,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_4 = np.reshape(log_p_mu_4,(64,64,24))[:,:,j:j+1]
std_f_vae_4 = (np.reshape(f_sigma_4,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_4 = np.reshape(att_sigma_4,(64,64,24))[:,:,j:j+1]
std_e_vae_4 = (np.reshape(np.sqrt((np.exp(e_sigma_4**2)-1)*np.exp(2*e_mu_4 + e_sigma_4**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_4 = np.reshape(log_s_sigma_4,(64,64,24))[:,:,j:j+1]
std_log_p_vae_4 = np.reshape(log_p_sigma_4,(64,64,24))[:,:,j:j+1]
mean_s_vae_4 = np.exp(mean_log_s_vae_4)
mean_s_vae_4[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_4 = np.exp(mean_log_p_vae_4)
mean_p_vae_4[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_4[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_4[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_4[np.where(mean_f_vae_4 < thr)] = 0
std_att_vae_4[np.where(mean_f_vae_4 < thr)] = 0



pred_encoder_5 = vae.encoder(x_data_5[:,0:9])
mu_5 = vae.fc_mu(pred_encoder_5)
sigma_5 = torch.exp(vae.fc_log_std(pred_encoder_5))
f_mu_5 = mu_5[:,0].detach().numpy()
att_mu_5 = mu_5[:,1].detach().numpy()
e_mu_5 = mu_5[:,2].detach().numpy()
log_s_mu_5 = mu_5[:,3].detach().numpy()
log_p_mu_5 = mu_5[:,4].detach().numpy()
f_sigma_5 = sigma_5[:,0].detach().numpy()
att_sigma_5 = sigma_5[:,1].detach().numpy()
e_sigma_5 = sigma_5[:,2].detach().numpy()
log_s_sigma_5 = sigma_5[:,3].detach().numpy()
log_p_sigma_5 = sigma_5[:,4].detach().numpy()
f_mu_5[idx_mask] = 0
att_mu_5[idx_mask] = 0
e_mu_5[idx_mask] = 0
log_s_mu_5[idx_mask] = 0 
log_p_mu_5[idx_mask] = 0  
f_sigma_5[idx_mask] = 0
att_sigma_5[idx_mask] = 0
e_sigma_5[idx_mask] = 0
log_s_sigma_5[idx_mask] = 0
log_p_sigma_5[idx_mask] = 0

mean_f_vae_5 = (np.reshape(f_mu_5,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_5 = np.reshape(att_mu_5,(64,64,24))[:,:,j:j+1]
mean_e_vae_5 = (np.reshape(np.exp(e_mu_5 + e_sigma_5**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_e_vae_5[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_log_s_vae_5 = np.reshape(log_s_mu_5,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_5 = np.reshape(log_p_mu_5,(64,64,24))[:,:,j:j+1]
std_f_vae_5 = (np.reshape(f_sigma_5,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_5 = np.reshape(att_sigma_5,(64,64,24))[:,:,j:j+1]
std_e_vae_5 = (np.reshape(np.sqrt((np.exp(e_sigma_5**2)-1)*np.exp(2*e_mu_5 + e_sigma_5**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_e_vae_5[np.where(mask[:,:,j:j+1] == 0)] = 0
std_log_s_vae_5 = np.reshape(log_s_sigma_5,(64,64,24))[:,:,j:j+1]
std_log_p_vae_5 = np.reshape(log_p_sigma_5,(64,64,24))[:,:,j:j+1]
mean_s_vae_5 = np.exp(mean_log_s_vae_5)
mean_s_vae_5[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_5 = np.exp(mean_log_p_vae_5)
mean_p_vae_5[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_5[np.where(mean_f_vae_5 < thr)] = 0
std_att_vae_5[np.where(mean_f_vae_5 < thr)] = 0





pred_encoder_6 = vae.encoder(x_data_6[:,0:9])
mu_6 = vae.fc_mu(pred_encoder_6)
sigma_6 = torch.exp(vae.fc_log_std(pred_encoder_6))
f_mu_6 = mu_6[:,0].detach().numpy()
att_mu_6 = mu_6[:,1].detach().numpy()
e_mu_6 = mu_6[:,2].detach().numpy()
log_s_mu_6 = mu_6[:,3].detach().numpy()
log_p_mu_6 = mu_6[:,4].detach().numpy()
f_sigma_6 = sigma_6[:,0].detach().numpy()
att_sigma_6 = sigma_6[:,1].detach().numpy()
e_sigma_6 = sigma_6[:,2].detach().numpy()
log_s_sigma_6 = sigma_6[:,3].detach().numpy()
log_p_sigma_6 = sigma_6[:,4].detach().numpy()
f_mu_6[idx_mask] = 0
att_mu_6[idx_mask] = 0
e_mu_6[idx_mask] = 0
log_s_mu_6[idx_mask] = 0 
log_p_mu_6[idx_mask] = 0  
f_sigma_6[idx_mask] = 0
att_sigma_6[idx_mask] = 0
e_sigma_6[idx_mask] = 0
log_s_sigma_6[idx_mask] = 0
log_p_sigma_6[idx_mask] = 0

mean_f_vae_6 = (np.reshape(f_mu_6,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_6 = np.reshape(att_mu_6,(64,64,24))[:,:,j:j+1]
mean_e_vae_6 = (np.reshape(np.exp(e_mu_6 + e_sigma_6**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_6 = np.reshape(log_s_mu_6,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_6 = np.reshape(log_p_mu_6,(64,64,24))[:,:,j:j+1]
std_f_vae_6 = (np.reshape(f_sigma_6,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_6 = np.reshape(att_sigma_6,(64,64,24))[:,:,j:j+1]
std_e_vae_6 = (np.reshape(np.sqrt((np.exp(e_sigma_6**2)-1)*np.exp(2*e_mu_6 + e_sigma_6**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_6 = np.reshape(log_s_sigma_6,(64,64,24))[:,:,j:j+1]
std_log_p_vae_6 = np.reshape(log_p_sigma_6,(64,64,24))[:,:,j:j+1]
mean_s_vae_6 = np.exp(mean_log_s_vae_6)
mean_s_vae_6[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_6 = np.exp(mean_log_p_vae_6)
mean_p_vae_6[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_6[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_6[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_6[np.where(mean_f_vae_6 < thr)] = 0
std_att_vae_6[np.where(mean_f_vae_6 < thr)] = 0





pred_encoder_7 = vae.encoder(x_data_7[:,0:9])
mu_7 = vae.fc_mu(pred_encoder_7)
sigma_7 = torch.exp(vae.fc_log_std(pred_encoder_7))
f_mu_7 = mu_7[:,0].detach().numpy()
att_mu_7 = mu_7[:,1].detach().numpy()
e_mu_7 = mu_7[:,2].detach().numpy()
log_s_mu_7 = mu_7[:,3].detach().numpy()
log_p_mu_7 = mu_7[:,4].detach().numpy()
f_sigma_7 = sigma_7[:,0].detach().numpy()
att_sigma_7 = sigma_7[:,1].detach().numpy()
e_sigma_7 = sigma_7[:,2].detach().numpy()
log_s_sigma_7 = sigma_7[:,3].detach().numpy()
log_p_sigma_7 = sigma_7[:,4].detach().numpy()
f_mu_7[idx_mask] = 0
att_mu_7[idx_mask] = 0
e_mu_7[idx_mask] = 0
log_s_mu_7[idx_mask] = 0 
log_p_mu_7[idx_mask] = 0  
f_sigma_7[idx_mask] = 0
att_sigma_7[idx_mask] = 0
e_sigma_7[idx_mask] = 0
log_s_sigma_7[idx_mask] = 0
log_p_sigma_7[idx_mask] = 0

mean_f_vae_7 = (np.reshape(f_mu_7,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_7 = np.reshape(att_mu_7,(64,64,24))[:,:,j:j+1]
mean_e_vae_7 = (np.reshape(np.exp(e_mu_7 + e_sigma_7**2/2),(64,64,24))*scaling)[:,:,j:j+1]
mean_log_s_vae_7 = np.reshape(log_s_mu_7,(64,64,24))[:,:,j:j+1]
mean_log_p_vae_7 = np.reshape(log_p_mu_7,(64,64,24))[:,:,j:j+1]
std_f_vae_7 = (np.reshape(f_sigma_7,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_7 = np.reshape(att_sigma_7,(64,64,24))[:,:,j:j+1]
std_e_vae_7 = (np.reshape(np.sqrt((np.exp(e_sigma_7**2)-1)*np.exp(2*e_mu_7 + e_sigma_7**2)),(64,64,24))*scaling)[:,:,j:j+1]
std_log_s_vae_7 = np.reshape(log_s_sigma_7,(64,64,24))[:,:,j:j+1]
std_log_p_vae_7 = np.reshape(log_p_sigma_7,(64,64,24))[:,:,j:j+1]
mean_s_vae_7 = np.exp(mean_log_s_vae_7)
mean_s_vae_7[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_p_vae_7 = np.exp(mean_log_p_vae_7)
mean_p_vae_7[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_e_vae_7[np.where(mask[:,:,j:j+1] == 0)] = 0
std_e_vae_7[np.where(mask[:,:,j:j+1] == 0)] = 0
mean_att_vae_7[np.where(mean_f_vae_7 < thr)] = 0
std_att_vae_7[np.where(mean_f_vae_7 < thr)] = 0









######################## VAE

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(f) Estimated Perfusion by VAE-like Framework, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(mean_f_vae_0*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(mean_f_vae_1*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im2_vae = axes[0,2].imshow(np.rot90(mean_f_vae_2*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im3_vae = axes[0,3].imshow(np.rot90(mean_f_vae_3*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(mean_f_vae_4*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(mean_f_vae_5*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(mean_f_vae_6*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(mean_f_vae_7*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
axes[1,0].set_title('Repeat 5',fontsize=10)
axes[1,1].set_title('Repeat 6',fontsize=10)
axes[1,2].set_title('Repeat 7',fontsize=10)
axes[1,3].set_title('Repeat 8',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,3].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[1,2].axis('off')
axes[1,3].axis('off')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.96, 0.05, 0.005, 0.80])
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
cbar = fig.colorbar(im3_vae, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_perfusion_single_repeat.png')
#plt.show()


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(h) Estimated ATT by VAE-like Framework, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(mean_att_vae_0,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(mean_att_vae_1,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im2_vae = axes[0,2].imshow(np.rot90(mean_att_vae_2,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im3_vae = axes[0,3].imshow(np.rot90(mean_att_vae_3,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(mean_att_vae_4,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(mean_att_vae_5,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(mean_att_vae_6,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(mean_att_vae_7,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
axes[1,0].set_title('Repeat 5',fontsize=10)
axes[1,1].set_title('Repeat 6',fontsize=10)
axes[1,2].set_title('Repeat 7',fontsize=10)
axes[1,3].set_title('Repeat 8',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,3].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[1,2].axis('off')
axes[1,3].axis('off')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.96, 0.05, 0.005, 0.80])
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
cbar = fig.colorbar(im3_vae, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_att_single_repeat.png')
#plt.show()



fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(f) 95% CI of Perfusion, VAE-like Framework, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(std_f_vae_0*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(std_f_vae_1*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im2_vae = axes[0,2].imshow(np.rot90(std_f_vae_2*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im3_vae = axes[0,3].imshow(np.rot90(std_f_vae_3*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(std_f_vae_4*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(std_f_vae_5*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(std_f_vae_6*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(std_f_vae_7*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
axes[1,0].set_title('Repeat 5',fontsize=10)
axes[1,1].set_title('Repeat 6',fontsize=10)
axes[1,2].set_title('Repeat 7',fontsize=10)
axes[1,3].set_title('Repeat 8',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,3].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[1,2].axis('off')
axes[1,3].axis('off')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.96, 0.05, 0.005, 0.80])
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
cbar = fig.colorbar(im3_vae, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_perfusion_std_single_repeat.png')
#plt.show()





fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(h) 95% CI of ATT, VAE-like Framework, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(std_att_vae_0,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(std_att_vae_1,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im2_vae = axes[0,2].imshow(np.rot90(std_att_vae_2,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im3_vae = axes[0,3].imshow(np.rot90(std_att_vae_3,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(std_att_vae_4,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(std_att_vae_5,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(std_att_vae_6,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(std_att_vae_7,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
axes[1,0].set_title('Repeat 5',fontsize=10)
axes[1,1].set_title('Repeat 6',fontsize=10)
axes[1,2].set_title('Repeat 7',fontsize=10)
axes[1,3].set_title('Repeat 8',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,3].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[1,2].axis('off')
axes[1,3].axis('off')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.96, 0.05, 0.005, 0.80])
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
cbar = fig.colorbar(im3_vae, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_att_std_single_repeat.png')
#plt.show()






mean_e_vae_0[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_1[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_2[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_3[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_4[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_5[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_6[np.where(mask[:,:,j:j+1] == 0 )] = 0
mean_e_vae_7[np.where(mask[:,:,j:j+1] == 0 )] = 0

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('Estimated Noise Parameter by VAE-like Framework, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90((mean_e_vae_0*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90((mean_e_vae_1*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im2_vae = axes[0,2].imshow(np.rot90((mean_e_vae_2*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im3_vae = axes[0,3].imshow(np.rot90((mean_e_vae_3*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90((mean_e_vae_4*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90((mean_e_vae_5*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90((mean_e_vae_6*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90((mean_e_vae_7*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
axes[1,0].set_title('Repeat 5',fontsize=10)
axes[1,1].set_title('Repeat 6',fontsize=10)
axes[1,2].set_title('Repeat 7',fontsize=10)
axes[1,3].set_title('Repeat 8',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,3].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[1,2].axis('off')
axes[1,3].axis('off')
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.96, 0.05, 0.005, 0.80])
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
cbar = fig.colorbar(im3_vae, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_Std of Error Term_single_repeat.png')
#plt.show()




