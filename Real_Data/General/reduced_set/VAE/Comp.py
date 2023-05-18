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
    def __init__(self, input_dim= 9, enc_out_dim=100, latent_dim=3):
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
    def tissue_signal(self,t, ftiss, delt, t1, t1b, tau):
        post_bolus = torch.greater(t, torch.add(tau, delt))
        during_bolus = torch.logical_and(torch.greater(t, delt), torch.logical_not(post_bolus))
        t1_app = 1 / (1 / t1 + 0.01 / 0.9)
        factor = 2 * t1_app * torch.exp(-delt / t1b)
        during_bolus_signal =  factor * (1 - torch.exp(-(t - delt) / t1_app))
        post_bolus_signal = factor * torch.exp(-(t - tau - delt) / t1_app) * (1 - torch.exp(-tau / t1_app))
        signal = torch.zeros(during_bolus_signal.shape)
        signal = torch.where(during_bolus, during_bolus_signal, signal.double())
        signal = torch.where(post_bolus, post_bolus_signal, signal.double())
        out = ftiss*signal
        return out





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
scaling = 100/(1090*0.85)
thr = 0.055

pred_encoder_0 = vae.encoder(x_data_0[:,0:9])
mu_0 = vae.fc_mu(pred_encoder_0)
sigma_0 = torch.exp(vae.fc_log_std(pred_encoder_0))
f_mu_0 = mu_0[:,0].detach().numpy()
att_mu_0 = mu_0[:,1].detach().numpy()
e_mu_0 = mu_0[:,2].detach().numpy()
f_sigma_0 = sigma_0[:,0].detach().numpy()
att_sigma_0 = sigma_0[:,1].detach().numpy()
e_sigma_0 = sigma_0[:,2].detach().numpy()
f_mu_0[idx_mask] = 0
att_mu_0[idx_mask] = 0
e_mu_0[idx_mask] = 0
f_sigma_0[idx_mask] = 0
att_sigma_0[idx_mask] = 0
e_sigma_0[idx_mask] = 0


mean_f_vae_0 = (np.reshape(f_mu_0,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_0 = np.reshape(att_mu_0,(64,64,24))[:,:,j:j+1]
mean_e_vae_0 = (np.reshape(np.exp(e_mu_0 + e_sigma_0**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_0 = (np.reshape(f_sigma_0,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_0 = np.reshape(att_sigma_0,(64,64,24))[:,:,j:j+1]
std_e_vae_0 = (np.reshape(np.sqrt((np.exp(e_sigma_0**2)-1)*np.exp(2*e_mu_0 + e_sigma_0**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_0[np.where(mean_f_vae_0 < thr)] = 0
std_att_vae_0[np.where(mean_f_vae_0 < thr)] = 0





pred_encoder_1 = vae.encoder(x_data_1[:,0:9])
mu_1 = vae.fc_mu(pred_encoder_1)
sigma_1 = torch.exp(vae.fc_log_std(pred_encoder_1))
f_mu_1 = mu_1[:,0].detach().numpy()
att_mu_1 = mu_1[:,1].detach().numpy()
e_mu_1 = mu_1[:,2].detach().numpy()
f_sigma_1 = sigma_1[:,0].detach().numpy()
att_sigma_1 = sigma_1[:,1].detach().numpy()
e_sigma_1 = sigma_1[:,2].detach().numpy()
f_mu_1[idx_mask] = 0
att_mu_1[idx_mask] = 0
e_mu_1[idx_mask] = 0
f_sigma_1[idx_mask] = 0
att_sigma_1[idx_mask] = 0
e_sigma_1[idx_mask] = 0

mean_f_vae_1 = (np.reshape(f_mu_1,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_1 = np.reshape(att_mu_1,(64,64,24))[:,:,j:j+1]
mean_e_vae_1 = (np.reshape(np.exp(e_mu_1 + e_sigma_1**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_1 = (np.reshape(f_sigma_1,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_1 = np.reshape(att_sigma_1,(64,64,24))[:,:,j:j+1]
std_e_vae_1 = (np.reshape(np.sqrt((np.exp(e_sigma_1**2)-1)*np.exp(2*e_mu_1 + e_sigma_1**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_1[np.where(mean_f_vae_1 < thr)] = 0
std_att_vae_1[np.where(mean_f_vae_1 < thr)] = 0



pred_encoder_2 = vae.encoder(x_data_2[:,0:9])
mu_2 = vae.fc_mu(pred_encoder_2)
sigma_2 = torch.exp(vae.fc_log_std(pred_encoder_2))
f_mu_2 = mu_2[:,0].detach().numpy()
att_mu_2 = mu_2[:,1].detach().numpy()
e_mu_2 = mu_2[:,2].detach().numpy()
f_sigma_2 = sigma_2[:,0].detach().numpy()
att_sigma_2 = sigma_2[:,1].detach().numpy()
e_sigma_2 = sigma_2[:,2].detach().numpy()
f_mu_2[idx_mask] = 0
att_mu_2[idx_mask] = 0
e_mu_2[idx_mask] = 0
f_sigma_2[idx_mask] = 0
att_sigma_2[idx_mask] = 0
e_sigma_2[idx_mask] = 0


mean_f_vae_2 = (np.reshape(f_mu_2,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_2 = np.reshape(att_mu_2,(64,64,24))[:,:,j:j+1]
mean_e_vae_2 = (np.reshape(np.exp(e_mu_2 + e_sigma_2**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_2 = (np.reshape(f_sigma_2,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_2 = np.reshape(att_sigma_2,(64,64,24))[:,:,j:j+1]
std_e_vae_2 = (np.reshape(np.sqrt((np.exp(e_sigma_2**2)-1)*np.exp(2*e_mu_2 + e_sigma_2**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_2[np.where(mean_f_vae_2 < thr)] = 0
std_att_vae_2[np.where(mean_f_vae_2 < thr)] = 0



pred_encoder_3 = vae.encoder(x_data_3[:,0:9])
mu_3 = vae.fc_mu(pred_encoder_3)
sigma_3 = torch.exp(vae.fc_log_std(pred_encoder_3))
f_mu_3 = mu_3[:,0].detach().numpy()
att_mu_3 = mu_3[:,1].detach().numpy()
e_mu_3 = mu_3[:,2].detach().numpy()
f_sigma_3 = sigma_3[:,0].detach().numpy()
att_sigma_3 = sigma_3[:,1].detach().numpy()
e_sigma_3 = sigma_3[:,2].detach().numpy()
f_mu_3[idx_mask] = 0
att_mu_3[idx_mask] = 0
e_mu_3[idx_mask] = 0
f_sigma_3[idx_mask] = 0
att_sigma_3[idx_mask] = 0
e_sigma_3[idx_mask] = 0

mean_f_vae_3 = (np.reshape(f_mu_3,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_3 = np.reshape(att_mu_3,(64,64,24))[:,:,j:j+1]
mean_e_vae_3 = (np.reshape(np.exp(e_mu_3 + e_sigma_3**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_3 = (np.reshape(f_sigma_3,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_3 = np.reshape(att_sigma_3,(64,64,24))[:,:,j:j+1]
std_e_vae_3 = (np.reshape(np.sqrt((np.exp(e_sigma_3**2)-1)*np.exp(2*e_mu_3 + e_sigma_3**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_3[np.where(mean_f_vae_3 < thr)] = 0
std_att_vae_3[np.where(mean_f_vae_3 < thr)] = 0




pred_encoder_4 = vae.encoder(x_data_4[:,0:9])
mu_4 = vae.fc_mu(pred_encoder_4)
sigma_4 = torch.exp(vae.fc_log_std(pred_encoder_4))
f_mu_4 = mu_4[:,0].detach().numpy()
att_mu_4 = mu_4[:,1].detach().numpy()
e_mu_4 = mu_4[:,2].detach().numpy()
f_sigma_4 = sigma_4[:,0].detach().numpy()
att_sigma_4 = sigma_4[:,1].detach().numpy()
e_sigma_4 = sigma_4[:,2].detach().numpy()
f_mu_4[idx_mask] = 0
att_mu_4[idx_mask] = 0
e_mu_4[idx_mask] = 0
f_sigma_4[idx_mask] = 0
att_sigma_4[idx_mask] = 0
e_sigma_4[idx_mask] = 0

mean_f_vae_4 = (np.reshape(f_mu_4,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_4 = np.reshape(att_mu_4,(64,64,24))[:,:,j:j+1]
mean_e_vae_4 = (np.reshape(np.exp(e_mu_4 + e_sigma_4**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_4 = (np.reshape(f_sigma_4,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_4 = np.reshape(att_sigma_4,(64,64,24))[:,:,j:j+1]
std_e_vae_4 = (np.reshape(np.sqrt((np.exp(e_sigma_4**2)-1)*np.exp(2*e_mu_4 + e_sigma_4**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_4[np.where(mean_f_vae_4 < thr)] = 0
std_att_vae_4[np.where(mean_f_vae_4 < thr)] = 0



pred_encoder_5 = vae.encoder(x_data_5[:,0:9])
mu_5 = vae.fc_mu(pred_encoder_5)
sigma_5 = torch.exp(vae.fc_log_std(pred_encoder_5))
f_mu_5 = mu_5[:,0].detach().numpy()
att_mu_5 = mu_5[:,1].detach().numpy()
e_mu_5 = mu_5[:,2].detach().numpy()
f_sigma_5 = sigma_5[:,0].detach().numpy()
att_sigma_5 = sigma_5[:,1].detach().numpy()
e_sigma_5 = sigma_5[:,2].detach().numpy()
f_mu_5[idx_mask] = 0
att_mu_5[idx_mask] = 0
e_mu_5[idx_mask] = 0
f_sigma_5[idx_mask] = 0
att_sigma_5[idx_mask] = 0
e_sigma_5[idx_mask] = 0


mean_f_vae_5 = (np.reshape(f_mu_5,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_5 = np.reshape(att_mu_5,(64,64,24))[:,:,j:j+1]
mean_e_vae_5 = (np.reshape(np.exp(e_mu_5 + e_sigma_5**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_5 = (np.reshape(f_sigma_5,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_5 = np.reshape(att_sigma_5,(64,64,24))[:,:,j:j+1]
std_e_vae_5 = (np.reshape(np.sqrt((np.exp(e_sigma_5**2)-1)*np.exp(2*e_mu_5 + e_sigma_5**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_5[np.where(mean_f_vae_5 < thr)] = 0
std_att_vae_5[np.where(mean_f_vae_5 < thr)] = 0





pred_encoder_6 = vae.encoder(x_data_6[:,0:9])
mu_6 = vae.fc_mu(pred_encoder_6)
sigma_6 = torch.exp(vae.fc_log_std(pred_encoder_6))
f_mu_6 = mu_6[:,0].detach().numpy()
att_mu_6 = mu_6[:,1].detach().numpy()
e_mu_6 = mu_6[:,2].detach().numpy()
f_sigma_6 = sigma_6[:,0].detach().numpy()
att_sigma_6 = sigma_6[:,1].detach().numpy()
e_sigma_6 = sigma_6[:,2].detach().numpy()
f_mu_6[idx_mask] = 0
att_mu_6[idx_mask] = 0
e_mu_6[idx_mask] = 0
f_sigma_6[idx_mask] = 0
att_sigma_6[idx_mask] = 0
e_sigma_6[idx_mask] = 0

mean_f_vae_6 = (np.reshape(f_mu_6,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_6 = np.reshape(att_mu_6,(64,64,24))[:,:,j:j+1]
mean_e_vae_6 = (np.reshape(np.exp(e_mu_6 + e_sigma_6**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_6 = (np.reshape(f_sigma_6,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_6 = np.reshape(att_sigma_6,(64,64,24))[:,:,j:j+1]
std_e_vae_6 = (np.reshape(np.sqrt((np.exp(e_sigma_6**2)-1)*np.exp(2*e_mu_6 + e_sigma_6**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_6[np.where(mean_f_vae_6 < thr)] = 0
std_att_vae_6[np.where(mean_f_vae_6 < thr)] = 0



pred_encoder_7 = vae.encoder(x_data_7[:,0:9])
mu_7 = vae.fc_mu(pred_encoder_7)
sigma_7 = torch.exp(vae.fc_log_std(pred_encoder_7))
f_mu_7 = mu_7[:,0].detach().numpy()
att_mu_7 = mu_7[:,1].detach().numpy()
e_mu_7 = mu_7[:,2].detach().numpy()
f_sigma_7 = sigma_7[:,0].detach().numpy()
att_sigma_7 = sigma_7[:,1].detach().numpy()
e_sigma_7 = sigma_7[:,2].detach().numpy()
f_mu_7[idx_mask] = 0
att_mu_7[idx_mask] = 0
e_mu_7[idx_mask] = 0
f_sigma_7[idx_mask] = 0
att_sigma_7[idx_mask] = 0
e_sigma_7[idx_mask] = 0


mean_f_vae_7 = (np.reshape(f_mu_7,(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_7 = np.reshape(att_mu_7,(64,64,24))[:,:,j:j+1]
mean_e_vae_7 = (np.reshape(np.exp(e_mu_7 + e_sigma_7**2/2),(64,64,24))*scaling)[:,:,j:j+1]
std_f_vae_7 = (np.reshape(f_sigma_7,(64,64,24))*scaling)[:,:,j:j+1]
std_att_vae_7 = np.reshape(att_sigma_7,(64,64,24))[:,:,j:j+1]
std_e_vae_7 = (np.reshape(np.sqrt((np.exp(e_sigma_7**2)-1)*np.exp(2*e_mu_7 + e_sigma_7**2)),(64,64,24))*scaling)[:,:,j:j+1]
mean_att_vae_7[np.where(mean_f_vae_7 < thr)] = 0
std_att_vae_7[np.where(mean_f_vae_7 < thr)] = 0









######################## VAE

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(b) Estimated Perfusion by VAE-like Framework, General KM, Reduced Data Sets', fontsize=16, y = 0.96)
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
fig.suptitle('(d) Estimated ATT by VAE-like Framework, General KM, Reduced Data Sets', fontsize=16, y = 0.96)
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
fig.suptitle('(b) 95% CI of Perfusion, VAE-like Framework, General KM, Reduced Data Sets', fontsize=16, y = 0.96)
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
fig.suptitle('(d) 95% CI of ATT, VAE-like Framework, General KM, Reduced Data Sets', fontsize=16, y = 0.96)
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
fig.suptitle('Estimated Noise Parameter by VAE-like framework, General KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(mean_e_vae_0*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(mean_e_vae_1*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im2_vae = axes[0,2].imshow(np.rot90(mean_e_vae_2*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im3_vae = axes[0,3].imshow(np.rot90(mean_e_vae_3*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_vae = axes[1,0].imshow(np.rot90(mean_e_vae_4*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im1_vae = axes[1,1].imshow(np.rot90(mean_e_vae_5*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im2_vae = axes[1,2].imshow(np.rot90(mean_e_vae_6*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im3_vae = axes[1,3].imshow(np.rot90(mean_e_vae_7*100,k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
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
plt.savefig('real_data_e_single_repeat.png')
#plt.show()


