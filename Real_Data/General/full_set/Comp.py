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








######################## AVB

outdir_0 = "avb_output"

j = 10
thr = 0.05

scaling = 100/(1090*0.85)
mask = nib.load('mask.nii.gz')
mask = np.array(mask.dataobj)

mean_f_avb = (nib.load("%s/mean_ftiss.nii.gz" % outdir_0).get_data()*scaling)[:,:,j:j+1]
mean_att_avb = (nib.load("%s/mean_delttiss.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_e_avb = (1/(nib.load("%s/noise_means.nii.gz" % outdir_0).get_data())*scaling)
mean_e_avb[np.where(mask == 0)] = 0
mean_e_avb = mean_e_avb[:,:,j:j+1]
std_f_avb = ((nib.load("%s/std_ftiss.nii.gz" % outdir_0).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb = ((nib.load("%s/std_delttiss.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]
std_att_avb[np.where(mean_f_avb < thr)] = 0
std_e_avb = (1/(nib.load("%s/noise_stdevs.nii.gz" % outdir_0).get_data()+10**-6)*scaling)
std_e_avb[np.where(mask == 0)] = 0




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


data = nib.load('diffdata.nii.gz')
data = np.array(data.dataobj)
data_ave = nib.load('diffdata_mean.nii.gz')
data_ave = np.array(data_ave.dataobj)
i = 3
set = [0+i,8+i,16+i,24+i,32+i,40+i]
data_single_pld = data[:,:,:,set]
mask = nib.load('mask.nii.gz')
mask = np.array(mask.dataobj)
mask_single_pld = mask
#asl_data = data_single_pld*(60/100)
asl_data = data_ave*(60/100)
asl_data[np.where(mask == 0)] = 0

########################## VAE

n = 64*64*24
signal = np.reshape(asl_data,(n,6))
tau = np.repeat(1.4,n)
t1 = np.repeat(1.3,n)
t1b = np.repeat(1.6,n)


x_data = np.zeros((n,3+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data[:,0:sig_dim] = signal
x_data[:,sig_dim] = np.reshape(tau,(n,))
x_data[:,sig_dim+1] = np.reshape(t1,(n,))
x_data[:,sig_dim+2] = np.reshape(t1b,(n,))
x_data = torch.from_numpy(x_data) 

idx_mask = np.where(np.reshape(mask,(n,)) == 0)

pred_encoder_inf = vae.encoder(x_data[:,0:9])
mu_inf = vae.fc_mu(pred_encoder_inf)
sigma_inf = torch.exp(vae.fc_log_std(pred_encoder_inf))
f_mu_inf = mu_inf[:,0].detach().numpy()
att_mu_inf = mu_inf[:,1].detach().numpy()
e_mu_inf = mu_inf[:,2].detach().numpy()
f_sigma_inf = sigma_inf[:,0].detach().numpy()
att_sigma_inf = sigma_inf[:,1].detach().numpy()
e_sigma_inf = sigma_inf[:,2].detach().numpy()
f_mu_inf[idx_mask] = 0
att_mu_inf[idx_mask] = 0
e_mu_inf[idx_mask] = 0
f_sigma_inf[idx_mask] = 0
att_sigma_inf[idx_mask] = 0
e_sigma_inf[idx_mask] = 0

mean_f = np.reshape(f_mu_inf,(64,64,24))
mean_att = np.reshape(att_mu_inf,(64,64,24))
mean_e = np.reshape(np.exp(e_mu_inf + e_sigma_inf**2/2),(64,64,24))
mean_e[np.where(mask == 0)] = 0
std_f = np.reshape(f_sigma_inf,(64,64,24))
std_att = np.reshape(att_sigma_inf,(64,64,24))
std_e = np.reshape(np.sqrt((np.exp(e_sigma_inf**2)-1)*np.exp(2*e_mu_inf + e_sigma_inf**2)),(64,64,24))
std_e[np.where(mask == 0)] = 0


mean_f_vae = (mean_f*100/(1090*0.85))[:,:,j:j+1]
std_f_vae = (std_f*100*(1.96*2)/(1090*0.85))[:,:,j:j+1]
mean_att_vae = mean_att[:,:,j:j+1]
std_att_vae = (std_att*(1.96*2))[:,:,j:j+1]
mean_e_vae = (mean_e*100/(1090*0.85))[:,:,j:j+1]
mean_att_vae[np.where(mean_f_vae < thr)] = 0
std_att_vae[np.where(mean_f_vae < thr)] = 0





fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(a) General KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(mean_f_avb*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(mean_f_vae*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
axes[0,0].set_title('Estimated Perfusion, aVB',fontsize=10)
axes[0,1].set_title('Estimated Perfusion, VAE-like',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(mean_att_avb,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(mean_att_vae,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
axes[1,0].set_title('Estimated ATT, aVB',fontsize=10)
axes[1,1].set_title('Estimated ATT, VAE-like',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
plt.colorbar(im0_vae, ax=axes[0,0])
plt.colorbar(im1_vae, ax=axes[0,1])
plt.colorbar(im0_avb, ax=axes[1,0])
plt.colorbar(im1_avb, ax=axes[1,1])
fig.tight_layout(pad=1.5)
plt.savefig('real_data_perfusion_ave.png')



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(a) General KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(std_f_avb*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(std_f_vae*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
axes[0,0].set_title('95% CI of Perfusion, aVB',fontsize=10)
axes[0,1].set_title('95% CI of Perfusion, VAE-like',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(std_att_avb,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(std_att_vae,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
axes[1,0].set_title('95% CI of ATT, aVB',fontsize=10)
axes[1,1].set_title('95% CI of ATT, VAE-like',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
plt.colorbar(im0_vae, ax=axes[0,0])
plt.colorbar(im1_vae, ax=axes[0,1])
plt.colorbar(im0_avb, ax=axes[1,0])
plt.colorbar(im1_avb, ax=axes[1,1])
fig.tight_layout(pad=1.5)
plt.savefig('real_data_ave_CI.png')
#plt.show()




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
#fig.set_size_inches(14, 6)
fig.suptitle('General KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0].imshow(np.rot90(mean_e_avb*100,k=1,axes=(0,1)), cmap = "hot",vmax=50,vmin=0)
im1_vae = axes[1].imshow(np.rot90(mean_e_vae*100,k=1,axes=(0,1)), cmap = "hot",vmax=50,vmin=0)
axes[0].set_title('Estimated Noise Parameter, aVB',fontsize=10)
axes[1].set_title('Estimated Noise Parameter, VAE-like',fontsize=10)
#im0_avb = axes[1,0].imshow(np.rot90(mean_att_avb,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
#im1_avb = axes[1,1].imshow(np.rot90(mean_att_vae,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
#axes[1,0].set_title('Estimated ATT, aVB',fontsize=10)
#axes[1,1].set_title('Estimated ATT, VAE-like',fontsize=10)
axes[0].axis('off')
axes[1].axis('off')
#axes[1,0].axis('off')
#axes[1,1].axis('off')
plt.colorbar(im0_vae, ax=axes[0])
plt.colorbar(im1_vae, ax=axes[1])
#plt.colorbar(im0_avb, ax=axes[1,0])
#plt.colorbar(im1_avb, ax=axes[1,1])
fig.tight_layout(pad=1.5)
plt.savefig('e_mean.png')
#plt.show()

