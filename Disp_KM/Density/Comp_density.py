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





class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int, #number of neurons in input layer
        planes: int, #number of neurons in hidden layer
    ) -> None:
        super().__init__() != 1
        self.Linear1 = nn.Linear(planes, planes)
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
path = 'Disp_KM/VAE_like_disp.pth'
vae = torch.load(path)

########################## Load Data
m = 0
k = 121
n = k+m

scale = 200
factor = 7.5

print(os.getcwd())
M0_ftiss = np.array(pd.read_csv('M0_ftiss.csv'))[m:n,1:]/scale
ftiss = M0_ftiss/factor
delttiss = np.array(pd.read_csv('delttiss.csv'))[m:n,1:]
e_log = np.array(pd.read_csv('e_log.csv'))[m:n,1:]
e_std = (np.exp(e_log) + 0.0000000001)/scale
tau = np.array(pd.read_csv('tau.csv'))[m:n,1:]
t1 = np.array(pd.read_csv('t1.csv'))[m:n,1:]
t1b = np.array(pd.read_csv('t1b.csv'))[m:n,1:]
log_s =  np.array(pd.read_csv('log_s.csv'))[m:n,1:]
log_p =  np.array(pd.read_csv('log_p.csv'))[m:n,1:]
signal = np.array(pd.read_csv('signal.csv'))[m:n,1:]/scale
signal_noise_inf = np.array(pd.read_csv('signal_noise_snr_10000000.csv'))[m:n,1:]/scale
signal_noise_10 = np.array(pd.read_csv('signal_noise_snr_100.csv'))[m:n,1:]/scale
signal_noise_5 = np.array(pd.read_csv('signal_noise_snr_50.csv'))[m:n,1:]/scale
signal_noise_2 = np.array(pd.read_csv('signal_noise_snr_25.csv'))[m:n,1:]/scale


x_data_inf = np.zeros((k,8+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_inf[:,0:sig_dim] = np.reshape(signal_noise_inf,(k,signal_noise_inf.shape[1]))
x_data_inf[:,sig_dim] = np.reshape(tau,(k,))
x_data_inf[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_inf[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_inf[:,sig_dim+3] = np.reshape(ftiss,(k,))
x_data_inf[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_inf[:,sig_dim+5] = np.reshape(e_std,(k,))
x_data_inf[:,sig_dim+6] = np.reshape(log_s,(k,))
x_data_inf[:,sig_dim+7] = np.reshape(log_p,(k,))
#x_data_inf[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_inf = torch.from_numpy(x_data_inf) 


x_data_10 = np.zeros((k,8+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_10[:,0:sig_dim] = np.reshape(signal_noise_10,(k,signal_noise_10.shape[1]))
x_data_10[:,sig_dim] = np.reshape(tau,(k,))
x_data_10[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_10[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_10[:,sig_dim+3] = np.reshape(ftiss,(k,))
x_data_10[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_10[:,sig_dim+5] = np.reshape(e_std,(k,))
x_data_10[:,sig_dim+6] = np.reshape(log_s,(k,))
x_data_10[:,sig_dim+7] = np.reshape(log_p,(k,))
#x_data_10[:,sig_dim+8:8+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_10= torch.from_numpy(x_data_10) 


x_data_5 = np.zeros((k,8+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_5[:,0:sig_dim] = np.reshape(signal_noise_5,(k,signal_noise_5.shape[1]))
x_data_5[:,sig_dim] = np.reshape(tau,(k,))
x_data_5[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_5[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_5[:,sig_dim+3] = np.reshape(ftiss,(k,))
x_data_5[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_5[:,sig_dim+5] = np.reshape(e_std,(k,))
x_data_5[:,sig_dim+6] = np.reshape(log_s,(k,))
x_data_5[:,sig_dim+7] = np.reshape(log_p,(k,))
x_data_5= torch.from_numpy(x_data_5) 


x_data_2 = np.zeros((k,8+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_2[:,0:sig_dim] = np.reshape(signal_noise_2,(k,signal_noise_2.shape[1]))
x_data_2[:,sig_dim] = np.reshape(tau,(k,))
x_data_2[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_2[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_2[:,sig_dim+3] = np.reshape(ftiss,(k,))
x_data_2[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_2[:,sig_dim+5] = np.reshape(e_std,(k,))
x_data_2[:,sig_dim+6] = np.reshape(log_s,(k,))
x_data_2[:,sig_dim+7] = np.reshape(log_p,(k,))
x_data_2= torch.from_numpy(x_data_2) 
















######################################################## Comparison
c = 7.5
######################## Run VAE
pred_encoder_inf = vae.encoder(x_data_inf[:,0:9])
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
mean_f_inf_vae = f_mu_inf/c
mean_att_inf_vae = att_mu_inf
mean_e_inf_vae = np.exp(e_mu_inf + e_sigma_inf**2/2)/c
mean_log_s_inf_vae = log_s_mu_inf
mean_log_p_inf_vae = log_p_mu_inf
std_f_inf_vae = f_sigma_inf/c
std_att_inf_vae = att_sigma_inf
std_e_inf_vae = np.sqrt((np.exp(e_sigma_inf**2)-1)*np.exp(2*e_mu_inf + e_sigma_inf**2))/c
std_log_s_inf_vae = log_s_sigma_inf
std_log_p_inf_vae = log_p_sigma_inf




pred_encoder_10 = vae.encoder(x_data_10[:,0:9])
mu_10 = vae.fc_mu(pred_encoder_10)
sigma_10 = torch.exp(vae.fc_log_std(pred_encoder_10))
f_mu_10 = mu_10[:,0].detach().numpy()
att_mu_10 = mu_10[:,1].detach().numpy()
e_mu_10 = mu_10[:,2].detach().numpy()
log_s_mu_10 = mu_10[:,3].detach().numpy()
log_p_mu_10 = mu_10[:,4].detach().numpy()
f_sigma_10 = sigma_10[:,0].detach().numpy()
att_sigma_10 = sigma_10[:,1].detach().numpy()
e_sigma_10 = sigma_10[:,2].detach().numpy()
log_s_sigma_10 = sigma_10[:,3].detach().numpy()
log_p_sigma_10 = sigma_10[:,4].detach().numpy()
mean_f_10_vae = f_mu_10 /c
mean_att_10_vae = att_mu_10 
mean_e_10_vae = np.exp(e_mu_10 + e_sigma_10**2/2)/c
mean_log_s_10_vae = log_s_mu_10
mean_log_p_10_vae = log_p_mu_10
std_f_10_vae = f_sigma_10/c
std_att_10_vae = att_sigma_10
std_e_10_vae = np.sqrt((np.exp(e_sigma_10**2)-1)*np.exp(2*e_mu_10 + e_sigma_10**2))/c
std_log_s_10_vae = log_s_sigma_10
std_log_p_10_vae = log_p_sigma_10





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
mean_f_5_vae = f_mu_5 /c
mean_att_5_vae = att_mu_5 
mean_e_5_vae = np.exp(e_mu_5 + e_sigma_5**2/2)/c
mean_log_s_5_vae = log_s_mu_5
mean_log_p_5_vae = log_p_mu_5
std_f_5_vae = f_sigma_5/c
std_att_5_vae = att_sigma_5
std_e_5_vae = np.sqrt((np.exp(e_sigma_5**2)-1)*np.exp(2*e_mu_5 + e_sigma_5**2))/c
std_log_s_5_vae = log_s_sigma_5
std_log_p_5_vae = log_p_sigma_5




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
mean_f_2_vae = f_mu_2 /c
mean_att_2_vae = att_mu_2 
mean_e_2_vae = np.exp(e_mu_2 + e_sigma_2**2/2)/c
mean_log_s_2_vae = log_s_mu_2
mean_log_p_2_vae = log_p_mu_2
std_f_2_vae = f_sigma_2/c
std_att_2_vae = att_sigma_2
std_e_2_vae = np.sqrt((np.exp(e_sigma_2**2)-1)*np.exp(2*e_mu_2 + e_sigma_2**2))/c
std_log_s_2_vae = log_s_sigma_2
std_log_p_2_vae = log_p_sigma_2





######################## AVB

ftiss_truth = pd.read_csv("M0_ftiss.csv")/(factor*scale)
delttiss_truth = pd.read_csv("delttiss.csv")
ftiss_truth = ftiss_truth.to_numpy()[:, 1:]
ftiss = ftiss_truth.reshape(k, 1, -1)
delttiss_truth = delttiss_truth.to_numpy()[:, 1:]
delttiss = delttiss_truth.reshape(k, 1, -1)
e_std = (pd.read_csv("e_std.csv").to_numpy()[:, 1:]).reshape(k, 1, -1)/(factor*scale)



outdir_inf = "avb_output/avb_km_snr_10000000"
outdir_10 = "avb_output/avb_km_snr_100"
outdir_5 = "avb_output/avb_km_snr_50"
outdir_2 = "avb_output/avb_km_snr_25"

n = k

mean_f_inf_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_inf).get_data(),(n,))/factor
mean_att_inf_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_inf).get_data(),(n,))
mean_e_inf_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_inf).get_data(),(n,)))/factor
std_f_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_inf).get_data(),(n,)))/factor
std_att_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_inf).get_data(),(n,)))
std_e_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_inf).get_data(),(n,)))/factor
mean_s_inf_avb = np.reshape(nib.load("%s/mean_s_native.nii.gz" % outdir_inf).get_data(),(n,))
mean_sp_inf_avb = np.reshape(nib.load("%s/mean_sp_native.nii.gz" % outdir_inf).get_data(),(n,))
std_s_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_s_native.nii.gz" % outdir_inf).get_data(),(n,)))
std_sp_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_sp_native.nii.gz" % outdir_inf).get_data(),(n,)))




mean_f_10_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_10).get_data(),(n,))/factor
mean_att_10_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_10).get_data(),(n,))
mean_e_10_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_10).get_data(),(n,)))/factor
std_f_10_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_10).get_data(),(n,)))/factor
std_att_10_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_10).get_data(),(n,)))
std_e_10_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_10).get_data(),(n,)))/factor
mean_s_10_avb = np.reshape(nib.load("%s/mean_s_native.nii.gz" % outdir_10).get_data(),(n,))
mean_sp_10_avb = np.reshape(nib.load("%s/mean_sp_native.nii.gz" % outdir_10).get_data(),(n,))
std_s_10_avb = np.sqrt(np.reshape(nib.load("%s/var_s_native.nii.gz" % outdir_10).get_data(),(n,)))
std_sp_10_avb = np.sqrt(np.reshape(nib.load("%s/var_sp_native.nii.gz" % outdir_10).get_data(),(n,)))



mean_f_5_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_5).get_data(),(n,))/factor
mean_att_5_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_5).get_data(),(n,))
mean_e_5_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_5).get_data(),(n,)))/factor
std_f_5_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_5).get_data(),(n,)))/factor
std_att_5_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_5).get_data(),(n,)))
std_e_5_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_5).get_data(),(n,)))/factor
mean_s_5_avb = np.reshape(nib.load("%s/mean_s_native.nii.gz" % outdir_5).get_data(),(n,))
mean_sp_5_avb = np.reshape(nib.load("%s/mean_sp_native.nii.gz" % outdir_5).get_data(),(n,))
std_s_5_avb = np.sqrt(np.reshape(nib.load("%s/var_s_native.nii.gz" % outdir_5).get_data(),(n,)))
std_sp_5_avb = np.sqrt(np.reshape(nib.load("%s/var_sp_native.nii.gz" % outdir_5).get_data(),(n,)))



mean_f_2_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_2).get_data(),(n,))/factor
mean_att_2_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_2).get_data(),(n,))
mean_e_2_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_2).get_data(),(n,)))/factor
std_f_2_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_2).get_data(),(n,)))/factor
std_att_2_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_2).get_data(),(n,)))
std_e_2_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_2).get_data(),(n,)))/factor
mean_s_2_avb = np.reshape(nib.load("%s/mean_s_native.nii.gz" % outdir_2).get_data(),(n,))
mean_sp_2_avb = np.reshape(nib.load("%s/mean_sp_native.nii.gz" % outdir_2).get_data(),(n,))
std_s_2_avb = np.sqrt(np.reshape(nib.load("%s/var_s_native.nii.gz" % outdir_2).get_data(),(n,)))
std_sp_2_avb = np.sqrt(np.reshape(nib.load("%s/var_sp_native.nii.gz" % outdir_2).get_data(),(n,)))


######################## MCMC
n = 121

ftiss_truth = pd.read_csv("M0_ftiss.csv")/(factor*scale)
delttiss_truth = pd.read_csv("delttiss.csv")
ftiss_truth = ftiss_truth.to_numpy()[:, 1:]
ftiss = ftiss_truth.reshape(n, 1, -1)
delttiss_truth = delttiss_truth.to_numpy()[:, 1:]
delttiss = delttiss_truth.reshape(n, 1, -1)
e_std = (pd.read_csv("e_std.csv").to_numpy()[:, 1:]).reshape(n, 1, -1)/(factor*scale)

nsamp = 10000

f_est_inf = np.array(pd.read_csv('MCMC_output/est_ftiss_inf.csv'))[0:nsamp,1:]/factor
att_est_inf = np.array(pd.read_csv('MCMC_output/est_delt_inf.csv'))[0:nsamp,1:]
e_est_inf = np.array(pd.read_csv('MCMC_output/est_e_inf.csv'))[0:nsamp,1:]/factor
log_s_est_inf = np.array(pd.read_csv('MCMC_output/est_log_s_inf.csv'))[0:nsamp,1:]
log_p_est_inf = np.array(pd.read_csv('MCMC_output/est_log_p_inf.csv'))[0:nsamp,1:]
f_est_10 = np.array(pd.read_csv('MCMC_output/est_ftiss_10.csv'))[0:nsamp,1:]/factor
att_est_10 = np.array(pd.read_csv('MCMC_output/est_delt_10.csv'))[0:nsamp,1:]
e_est_10 = np.array(pd.read_csv('MCMC_output/est_e_10.csv'))[0:nsamp,1:]/factor
log_s_est_10 = np.array(pd.read_csv('MCMC_output/est_log_s_10.csv'))[0:nsamp,1:]
log_p_est_10 = np.array(pd.read_csv('MCMC_output/est_log_p_10.csv'))[0:nsamp,1:]
f_est_5 = np.array(pd.read_csv('MCMC_output/est_ftiss_5.csv'))[0:nsamp,1:]/factor
att_est_5 = np.array(pd.read_csv('MCMC_output/est_delt_5.csv'))[0:nsamp,1:]
e_est_5 = np.array(pd.read_csv('MCMC_output/est_e_5.csv'))[0:nsamp,1:]/factor
log_s_est_5 = np.array(pd.read_csv('MCMC_output/est_log_s_5.csv'))[0:nsamp,1:]
log_p_est_5 = np.array(pd.read_csv('MCMC_output/est_log_p_5.csv'))[0:nsamp,1:]
f_est_2 = np.array(pd.read_csv('MCMC_output/est_ftiss_2.csv'))[0:nsamp,1:]/factor
att_est_2 = np.array(pd.read_csv('MCMC_output/est_delt_2.csv'))[0:nsamp,1:]
e_est_2 = np.array(pd.read_csv('MCMC_output/est_e_2.csv'))[0:nsamp,1:]/factor
log_s_est_2 = np.array(pd.read_csv('MCMC_output/est_log_s_2.csv'))[0:nsamp,1:]
log_p_est_2 = np.array(pd.read_csv('MCMC_output/est_log_p_2.csv'))[0:nsamp,1:]


mean_f_inf_mcmc = np.mean(f_est_inf, axis = 0).reshape((n,))
mean_att_inf_mcmc = np.mean(att_est_inf, axis = 0).reshape((n,))
mean_e_inf_mcmc = np.mean(e_est_inf, axis = 0).reshape((n,))
mean_log_s_inf_mcmc = np.mean(log_s_est_inf, axis = 0).reshape((n,))
mean_log_p_inf_mcmc = np.mean(log_p_est_inf, axis = 0).reshape((n,))
std_e_inf_mcmc = np.quantile(e_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_inf, 0.025, axis=0).reshape((n,))
std_att_inf_mcmc = np.quantile(att_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_inf, 0.025, axis=0).reshape((n,))
std_f_inf_mcmc = np.quantile(f_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_inf, 0.025, axis=0).reshape((n,))
std_log_s_inf_mcmc = np.quantile(log_s_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(log_s_est_inf, 0.025, axis=0).reshape((n,))
std_log_p_inf_mcmc = np.quantile(log_p_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(log_p_est_inf, 0.025, axis=0).reshape((n,))

mean_f_10_mcmc = np.mean(f_est_10, axis = 0).reshape((n,))
mean_att_10_mcmc = np.mean(att_est_10, axis = 0).reshape((n,))
mean_e_10_mcmc = np.mean(e_est_10, axis = 0).reshape((n,))
mean_log_s_10_mcmc = np.mean(log_s_est_10, axis = 0).reshape((n,))
mean_log_p_10_mcmc = np.mean(log_p_est_10, axis = 0).reshape((n,))
std_e_10_mcmc = np.quantile(e_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_10, 0.025, axis=0).reshape((n,))
std_att_10_mcmc = np.quantile(att_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_10, 0.025, axis=0).reshape((n,))
std_f_10_mcmc = np.quantile(f_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_10, 0.025, axis=0).reshape((n,))
std_log_s_10_mcmc = np.quantile(log_s_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(log_s_est_10, 0.025, axis=0).reshape((n,))
std_log_p_10_mcmc = np.quantile(log_p_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(log_p_est_10, 0.025, axis=0).reshape((n,))

mean_f_5_mcmc = np.mean(f_est_5, axis = 0).reshape((n,))
mean_att_5_mcmc = np.mean(att_est_5, axis = 0).reshape((n,))
mean_e_5_mcmc = np.mean(e_est_5, axis = 0).reshape((n,))
mean_log_s_5_mcmc = np.mean(log_s_est_5, axis = 0).reshape((n,))
mean_log_p_5_mcmc = np.mean(log_p_est_5, axis = 0).reshape((n,))
std_e_5_mcmc = np.quantile(e_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_5, 0.025, axis=0).reshape((n,))
std_att_5_mcmc = np.quantile(att_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_5, 0.025, axis=0).reshape((n,))
std_f_5_mcmc = np.quantile(f_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_5, 0.025, axis=0).reshape((n,))
std_log_s_5_mcmc = np.quantile(log_s_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(log_s_est_5, 0.025, axis=0).reshape((n,))
std_log_p_5_mcmc = np.quantile(log_p_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(log_p_est_5, 0.025, axis=0).reshape((n,))

mean_f_2_mcmc = np.mean(f_est_2, axis = 0).reshape((n,))
mean_att_2_mcmc = np.mean(att_est_2, axis = 0).reshape((n,))
mean_e_2_mcmc = np.mean(e_est_2, axis = 0).reshape((n,))
mean_log_s_2_mcmc = np.mean(log_s_est_2, axis = 0).reshape((n,))
mean_log_p_2_mcmc = np.mean(log_p_est_2, axis = 0).reshape((n,))
std_e_2_mcmc = np.quantile(e_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_2, 0.025, axis=0).reshape((n,))
std_att_2_mcmc = np.quantile(att_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_2, 0.025, axis=0).reshape((n,))
std_f_2_mcmc = np.quantile(f_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_2, 0.025, axis=0).reshape((n,))
std_log_s_2_mcmc = np.quantile(log_s_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(log_s_est_2, 0.025, axis=0).reshape((n,))
std_log_p_2_mcmc = np.quantile(log_p_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(log_p_est_2, 0.025, axis=0).reshape((n,))




######## Heatmap 
####### a = 1, B = 1
x = np.linspace(0,3,51)
y = np.linspace(0,3,51)
int_x = x[1]-x[0]
int_y = y[1]-y[0]

###### 1) VAE

chk_x = 56
chk_y = 110


dist_vae_inf = multivariate_normal(mean=[mean_f_inf_vae[chk_x,], mean_att_inf_vae[chk_y,]], cov=[[std_f_inf_vae[chk_x,],0],[0,std_att_inf_vae[chk_y,]]])
hmp_vae_inf = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_inf[i,j] = dist_vae_inf.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_inf.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_inf.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_inf.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_vae_10 = multivariate_normal(mean=[mean_f_10_vae[chk_x,], mean_att_10_vae[chk_y,]], cov=[[std_f_10_vae[chk_x,],0],[0,std_att_10_vae[chk_y,]]])
hmp_vae_10 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_10[i,j] = dist_vae_10.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_10.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_10.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_10.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_vae_5 = multivariate_normal(mean=[mean_f_5_vae[chk_x,], mean_att_5_vae[chk_y,]], cov=[[std_f_5_vae[chk_x,],0],[0,std_att_5_vae[chk_y,]]])
hmp_vae_5 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_5[i,j] = dist_vae_5.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_5.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_5.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_5.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_vae_2 = multivariate_normal(mean=[mean_f_2_vae[chk_x,], mean_att_2_vae[chk_y,]], cov=[[std_f_2_vae[chk_x,],0],[0,std_att_2_vae[chk_y,]]])
hmp_vae_2 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_2[i,j] = dist_vae_2.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_2.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_2.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_2.cdf([x[i,]-int_x/2,y[j,]-int_y/2])





dist_avb_inf = multivariate_normal(mean=[mean_f_inf_avb[chk_x,], mean_att_inf_avb[chk_y,]], cov=[[std_f_inf_avb[chk_x,],0],[0,std_att_inf_avb[chk_y,]]])
hmp_avb_inf = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_inf[i,j] = dist_avb_inf.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_inf.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_inf.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_inf.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_avb_10 = multivariate_normal(mean=[mean_f_10_avb[chk_x,], mean_att_10_avb[chk_y,]], cov=[[std_f_10_avb[chk_x,],0],[0,std_att_10_avb[chk_y,]]])
hmp_avb_10 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_10[i,j] = dist_avb_10.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_10.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_10.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_10.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_avb_5 = multivariate_normal(mean=[mean_f_5_avb[chk_x,], mean_att_5_avb[chk_y,]], cov=[[std_f_5_avb[chk_x,],0],[0,std_att_5_avb[chk_y,]]])
hmp_avb_5 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_5[i,j] = dist_avb_5.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_5.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_5.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_5.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_avb_2 = multivariate_normal(mean=[mean_f_2_avb[chk_x,], mean_att_2_avb[chk_y,]], cov=[[std_f_2_avb[chk_x,],0],[0,std_att_2_avb[chk_y,]]])
hmp_avb_2 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_2[i,j] = dist_avb_2.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_2.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_2.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_2.cdf([x[i,]-int_x/2,y[j,]-int_y/2])




dist_f_mcmc_inf = f_est_inf[:,chk_x]
dist_att_mcmc_inf = att_est_inf[:,chk_y]
hmp_mcmc_inf = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_f_mcmc_inf>x[i]-int_x/2,dist_f_mcmc_inf<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_att_mcmc_inf>y[j]-int_y/2,dist_att_mcmc_inf<=y[j]+int_y/2)
        hmp_mcmc_inf[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp

dist_f_mcmc_10 = f_est_10[:,chk_x]
dist_att_mcmc_10 = att_est_10[:,chk_y]
hmp_mcmc_10 = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_f_mcmc_10>x[i]-int_x/2,dist_f_mcmc_10<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_att_mcmc_10>y[j]-int_y/2,dist_att_mcmc_10<=y[j]+int_y/2)
        hmp_mcmc_10[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp


dist_f_mcmc_5 = f_est_5[:,chk_x]
dist_att_mcmc_5 = att_est_5[:,chk_y]
hmp_mcmc_5 = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_f_mcmc_5>x[i]-int_x/2,dist_f_mcmc_5<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_att_mcmc_5>y[j]-int_y/2,dist_att_mcmc_5<=y[j]+int_y/2)
        hmp_mcmc_5[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp

dist_f_mcmc_2 = f_est_2[:,chk_x]
dist_att_mcmc_2 = att_est_2[:,chk_y]
hmp_mcmc_2 = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_f_mcmc_2>x[i]-int_x/2,dist_f_mcmc_2<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_att_mcmc_2>y[j]-int_y/2,dist_att_mcmc_2<=y[j]+int_y/2)
        hmp_mcmc_2[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp





fig, axes = plt.subplots(nrows=3, ncols=4)
fig.set_size_inches(10, 10)
fig.suptitle('Estimated Density Plot for Perfusion = 0.6 ml/g/min, ATT = 1.5 s')
extent = [x.min(), x.max(), y.min(), y.max()]
im0_vae = axes[0,0].imshow(hmp_vae_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_vae = axes[0,1].imshow(hmp_vae_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_vae = axes[0,2].imshow(hmp_vae_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_vae = axes[0,3].imshow(hmp_vae_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[0,0].set_title('VAE-like, SNR = INF \n Dispersion KM',fontsize=8)
axes[0,1].set_title('VAE-like, SNR = 10 \n Dispersion KM',fontsize=8)
axes[0,2].set_title('VAE-like, SNR = 5 \n Dispersion KM',fontsize=8)
axes[0,3].set_title('VAE-like, SNR = 2 \n Dispersion KM',fontsize=8)
im0_avb = axes[1,0].imshow(hmp_avb_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_avb = axes[1,1].imshow(hmp_avb_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_avb = axes[1,2].imshow(hmp_avb_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_avb = axes[1,3].imshow(hmp_avb_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[1,0].set_title('aVB, SNR = INF \n Dispersion KM',fontsize=8)
axes[1,1].set_title('aVB, SNR = 10 \n Dispersion KM',fontsize=8)
axes[1,2].set_title('aVB, SNR = 5 \n Dispersion KM',fontsize=8)
axes[1,3].set_title('aVB, SNR = 2 \n Dispersion KM',fontsize=8)
im0_mcmc = axes[2,0].imshow(hmp_mcmc_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_mcmc = axes[2,1].imshow(hmp_mcmc_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_mcmc = axes[2,2].imshow(hmp_mcmc_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_mcmc = axes[2,3].imshow(hmp_mcmc_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[2,0].set_title('MCMC, SNR = INF \n Dispersion KM',fontsize=8)
axes[2,1].set_title('MCMC, SNR = 10 \n Dispersion KM',fontsize=8)
axes[2,2].set_title('MCMC, SNR = 5 \n Dispersion KM',fontsize=8)
axes[2,3].set_title('MCMC, SNR = 2 \n Dispersion KM',fontsize=8)
fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.82, 0.32, 0.01, 0.40])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
#fig.colorbar(im3_vae, cax=cbar_ax)
fig.tight_layout(pad=0.1)
plt.savefig('density_disp_km.png')
#plt.show()




