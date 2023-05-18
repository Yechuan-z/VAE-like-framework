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
thr = 0.06

scaling = 200/(1090*0.85)
mask = nib.load('mask.nii.gz')
mask = np.array(mask.dataobj)

mean_f_avb = (nib.load("%s/mean_ftiss.nii.gz" % outdir_0).get_data()*scaling)[:,:,j:j+1]
mean_att_avb = (nib.load("%s/mean_delttiss.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_e_avb = (1/(nib.load("%s/noise_means.nii.gz" % outdir_0).get_data())*scaling)
mean_e_avb[np.where(mask == 0)] = 0
mean_e_avb = mean_e_avb[:,:,j:j+1]
std_f_avb = ((nib.load("%s/std_ftiss.nii.gz" % outdir_0).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb = ((nib.load("%s/std_delttiss.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb = (1/(nib.load("%s/noise_stdevs.nii.gz" % outdir_0).get_data()+10**-6)*scaling)
std_e_avb[np.where(mask == 0)] = 0
std_e_avb = std_e_avb[:,:,j:j+1]
mean_s_avb = (nib.load("%s/mean_disp1.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_sp_avb = (nib.load("%s/mean_disp2.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_p_avb = mean_sp_avb/(mean_s_avb+10**-6)
std_s_avb = ((nib.load("%s/std_disp1.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb = ((nib.load("%s/std_disp2.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]



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
asl_data = data_ave*(60/200)
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
log_s_mu_inf = mu_inf[:,3].detach().numpy()
log_p_mu_inf = mu_inf[:,4].detach().numpy()
f_sigma_inf = sigma_inf[:,0].detach().numpy()
att_sigma_inf = sigma_inf[:,1].detach().numpy()
e_sigma_inf = sigma_inf[:,2].detach().numpy()
log_s_sigma_inf = sigma_inf[:,3].detach().numpy()
log_p_sigma_inf = sigma_inf[:,4].detach().numpy()

f_mu_inf[idx_mask] = 0
att_mu_inf[idx_mask] = 0
e_mu_inf[idx_mask] = 0
log_s_mu_inf[idx_mask] = 0
log_p_mu_inf[idx_mask] = 0
f_sigma_inf[idx_mask] = -10**6
att_sigma_inf[idx_mask] = 0
e_sigma_inf[idx_mask] = 0
log_s_sigma_inf[idx_mask] = 0
log_p_sigma_inf[idx_mask] = 0

mean_f = np.reshape(f_mu_inf,(64,64,24))
mean_att = np.reshape(att_mu_inf,(64,64,24))
mean_e = np.reshape(np.exp(e_mu_inf + e_sigma_inf**2/2),(64,64,24))
mean_e[np.where(mask == 0)] = 0
mean_s = np.exp(np.reshape(log_s_mu_inf,(64,64,24)))
mean_s[np.where(mask == 0)] = 0
mean_p = np.exp(np.reshape(log_p_mu_inf,(64,64,24)))
mean_p[np.where(mask == 0)] = 0
std_f = np.reshape(f_sigma_inf,(64,64,24))
std_att = np.reshape(att_sigma_inf,(64,64,24))
std_e = np.reshape(np.sqrt((np.exp(e_sigma_inf**2)-1)*np.exp(2*e_mu_inf + e_sigma_inf**2)),(64,64,24))
std_e[np.where(mask == 0)] = 0
std_s = np.exp(np.reshape(log_s_sigma_inf,(64,64,24))*(1.96*2))
std_s[np.where(mask == 0)] = 0
std_p = np.exp(np.reshape(log_p_sigma_inf,(64,64,24))*(1.96*2))
std_p[np.where(mask == 0)] = 0


mean_f_vae = (mean_f*200/(1090*0.85))[:,:,j:j+1]
mean_att_vae = mean_att[:,:,j:j+1]
mean_e_vae = (mean_e*200/(1090*0.85))[:,:,j:j+1]
mean_s_vae = mean_s[:,:,j:j+1]
mean_p_vae = mean_p[:,:,j:j+1]
std_f_vae = (std_f*200*(1.96*2)/(1090*0.85))[:,:,j:j+1]
std_att_vae = (std_att*(1.96*2))[:,:,j:j+1]
std_e_vae = (std_e*200/(1090*0.85))[:,:,j:j+1]
std_s_vae = std_s
std_p_vae = std_p
mean_att_vae[np.where(mean_f_vae < thr)] = 0
std_att_vae[np.where(mean_f_vae < thr)] = 0






fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(b) Dispersion KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
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
#plt.show()







fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(b) Dispersion KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
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







fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('Dispersion KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0,0].imshow(np.rot90(mean_e_avb,k=1,axes=(0,1)), cmap = "hot",vmax=100,vmin=0)
im1_vae = axes[0,1].imshow(np.rot90(mean_e_vae,k=1,axes=(0,1)), cmap = "hot",vmax=100,vmin=0)
axes[1,0].set_title('Std of Error Term, aVB',fontsize=10)
axes[1,1].set_title('Std of Error Term, VAE-like',fontsize=10)
im0_vae = axes[1,0].imshow(np.rot90(mean_s_avb,k=1,axes=(0,1)), cmap = "hot",vmax=5,vmin=0)
im1_vae = axes[1,1].imshow(np.rot90(mean_s_vae,k=1,axes=(0,1)), cmap = "hot",vmax=5,vmin=0)
axes[1,0].set_title('Estimated Sharpness, aVB',fontsize=10)
axes[1,1].set_title('Estimated Sharpness, VAE-like',fontsize=10)
im0_avb = axes[2,0].imshow(np.rot90(mean_p_avb,k=1,axes=(0,1)), cmap = "hot",vmax=1,vmin=0)
im1_avb = axes[2,1].imshow(np.rot90(mean_p_vae,k=1,axes=(0,1)), cmap = "hot",vmax=1,vmin=0)
axes[2,0].set_title('Estimated Time-to-peak, aVB',fontsize=10)
axes[2,1].set_title('Estimated Time-to-peak, VAE-like',fontsize=10)
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[2,0].axis('off')
axes[2,1].axis('off')
plt.colorbar(im0_vae, ax=axes[0,0])
plt.colorbar(im1_vae, ax=axes[0,1])
plt.colorbar(im0_avb, ax=axes[1,0])
plt.colorbar(im1_avb, ax=axes[1,1])
plt.colorbar(im0_avb, ax=axes[2,0])
plt.colorbar(im1_avb, ax=axes[2,1])
fig.tight_layout(pad=1.5)
plt.savefig('real_data_esp_ave.png')
#plt.show()




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
#fig.set_size_inches(14, 6)
fig.suptitle('Dispersion KM, Averaged Real Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_vae = axes[0].imshow(np.rot90(mean_e_avb*100,k=1,axes=(0,1)), cmap = "hot",vmax=50,vmin=0)
im1_vae = axes[1].imshow(np.rot90(mean_e_vae*100,k=1,axes=(0,1)), cmap = "hot",vmax=50,vmin=0)
axes[0].set_title('Estimated Noise Parameter, aVB',fontsize=10)
axes[1].set_title('Estimated Noise Parameter, VAE-like',fontsize=10)
axes[0].axis('off')
axes[1].axis('off')
plt.colorbar(im0_vae, ax=axes[0])
plt.colorbar(im1_vae, ax=axes[1])
fig.tight_layout(pad=1.5)
plt.savefig('e_mean.png')

