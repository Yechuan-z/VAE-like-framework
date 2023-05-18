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
    def tissue_signal(self,t, M0_ftiss, delt, t1, t1b, tau):
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








########################## Load VAE
path = 'General_KM/VAE_like_general.pth'
vae = torch.load(path)


########################## Load Data
m = 0
k = 121
n = k+m

print(os.getcwd())
scale = 100
M0_ftiss = np.array(pd.read_csv('M0_ftiss.csv'))[m:n,1:]/scale
delttiss = np.array(pd.read_csv('delttiss.csv'))[m:n,1:]
e_log = np.array(pd.read_csv('e_log.csv'))[m:n,1:]
e_std = (np.exp(e_log) + 0.0000000001)/scale
tau = np.array(pd.read_csv('tau.csv'))[m:n,1:]
t1 = np.array(pd.read_csv('t1.csv'))[m:n,1:]
t1b = np.array(pd.read_csv('t1b.csv'))[m:n,1:]
signal = np.array(pd.read_csv('signal.csv'))[m:n,1:]/scale
signal_noise_inf = np.array(pd.read_csv('signal_noise_snr_10000000.csv'))[m:n,1:]/scale
signal_noise_10 = np.array(pd.read_csv('signal_noise_snr_100.csv'))[m:n,1:]/scale
signal_noise_5 = np.array(pd.read_csv('signal_noise_snr_50.csv'))[m:n,1:]/scale
signal_noise_2 = np.array(pd.read_csv('signal_noise_snr_25.csv'))[m:n,1:]/scale


x_data_inf = np.zeros((k,6+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_inf[:,0:sig_dim] = np.reshape(signal_noise_inf,(k,signal_noise_inf.shape[1]))
x_data_inf[:,sig_dim] = np.reshape(tau,(k,))
x_data_inf[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_inf[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_inf[:,sig_dim+3] = np.reshape(M0_ftiss,(k,))
x_data_inf[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_inf[:,sig_dim+5] = np.reshape(np.repeat(0,k).reshape(k,1),(k,))
#x_data[:,sig_dim+6:sig_dim+12] = np.reshape(signal,(k,signal.shape[1]))
x_data_inf = torch.from_numpy(x_data_inf) 


x_data_10 = np.zeros((k,6+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_10[:,0:sig_dim] = np.reshape(signal_noise_10,(k,signal_noise_10.shape[1]))
x_data_10[:,sig_dim] = np.reshape(tau,(k,))
x_data_10[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_10[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_10[:,sig_dim+3] = np.reshape(M0_ftiss,(k,))
x_data_10[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_10[:,sig_dim+5] = np.reshape(np.repeat(0.6/10,k).reshape(k,1),(k,))
x_data_10= torch.from_numpy(x_data_10) 


x_data_5 = np.zeros((k,6+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_5[:,0:sig_dim] = np.reshape(signal_noise_5,(k,signal_noise_5.shape[1]))
x_data_5[:,sig_dim] = np.reshape(tau,(k,))
x_data_5[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_5[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_5[:,sig_dim+3] = np.reshape(M0_ftiss,(k,))
x_data_5[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_5[:,sig_dim+5] = np.reshape(np.repeat(0.6/5,k).reshape(k,1),(k,))
x_data_5= torch.from_numpy(x_data_5) 


x_data_2 = np.zeros((k,6+signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_2[:,0:sig_dim] = np.reshape(signal_noise_2,(k,signal_noise_2.shape[1]))
x_data_2[:,sig_dim] = np.reshape(tau,(k,))
x_data_2[:,sig_dim+1] = np.reshape(t1,(k,))
x_data_2[:,sig_dim+2] = np.reshape(t1b,(k,))
x_data_2[:,sig_dim+3] = np.reshape(M0_ftiss,(k,))
x_data_2[:,sig_dim+4] = np.reshape(delttiss,(k,))
x_data_2[:,sig_dim+5] = np.reshape(np.repeat(0.6/2.5,k).reshape(k,1),(k,))
x_data_2= torch.from_numpy(x_data_2) 











######################################################## Comparison
c = 15
######################## Run VAE
pred_encoder_inf = vae.encoder(x_data_inf[:,0:9])
mu_inf = vae.fc_mu(pred_encoder_inf)
sigma_inf = torch.exp(vae.fc_log_std(pred_encoder_inf))
f_mu_inf = mu_inf[:,0].detach().numpy()
att_mu_inf = mu_inf[:,1].detach().numpy()
e_mu_inf = mu_inf[:,2].detach().numpy()
f_sigma_inf = sigma_inf[:,0].detach().numpy()
att_sigma_inf = sigma_inf[:,1].detach().numpy()
e_sigma_inf = sigma_inf[:,2].detach().numpy()
mean_f_inf_vae = f_mu_inf/c
mean_att_inf_vae = att_mu_inf
mean_e_inf_vae = np.exp(e_mu_inf + e_sigma_inf**2/2)/c
std_f_inf_vae = f_sigma_inf/c
std_att_inf_vae = att_sigma_inf
std_e_inf_vae = np.sqrt((np.exp(e_sigma_inf**2)-1)*np.exp(2*e_mu_inf + e_sigma_inf**2))/c


pred_encoder_10 = vae.encoder(x_data_10[:,0:9])
mu_10 = vae.fc_mu(pred_encoder_10)
sigma_10 = torch.exp(vae.fc_log_std(pred_encoder_10))
f_mu_10 = mu_10[:,0].detach().numpy()
att_mu_10 = mu_10[:,1].detach().numpy()
e_mu_10 = mu_10[:,2].detach().numpy()
f_sigma_10 = sigma_10[:,0].detach().numpy()
att_sigma_10 = sigma_10[:,1].detach().numpy()
e_sigma_10 = sigma_10[:,2].detach().numpy()
mean_f_10_vae = f_mu_10 /c
mean_att_10_vae = att_mu_10 
mean_e_10_vae = np.exp(e_mu_10 + e_sigma_10**2/2)/c
std_f_10_vae = f_sigma_10/c
std_att_10_vae = att_sigma_10
std_e_10_vae = np.sqrt((np.exp(e_sigma_10**2)-1)*np.exp(2*e_mu_10 + e_sigma_10**2))/c


pred_encoder_5 = vae.encoder(x_data_5[:,0:9])
mu_5 = vae.fc_mu(pred_encoder_5)
sigma_5 = torch.exp(vae.fc_log_std(pred_encoder_5))
f_mu_5 = mu_5[:,0].detach().numpy()
att_mu_5 = mu_5[:,1].detach().numpy()
e_mu_5 = mu_5[:,2].detach().numpy()
f_sigma_5 = sigma_5[:,0].detach().numpy()
att_sigma_5 = sigma_5[:,1].detach().numpy()
e_sigma_5 = sigma_5[:,2].detach().numpy()
mean_f_5_vae = f_mu_5 /c
mean_att_5_vae = att_mu_5 
mean_e_5_vae = np.exp(e_mu_5 + e_sigma_5**2/2)/c
std_f_5_vae = f_sigma_5/c
std_att_5_vae = att_sigma_5
std_e_5_vae = np.sqrt((np.exp(e_sigma_5**2)-1)*np.exp(2*e_mu_5 + e_sigma_5**2))/c


pred_encoder_2 = vae.encoder(x_data_2[:,0:9])
mu_2 = vae.fc_mu(pred_encoder_2)
sigma_2 = torch.exp(vae.fc_log_std(pred_encoder_2))
f_mu_2 = mu_2[:,0].detach().numpy()
att_mu_2 = mu_2[:,1].detach().numpy()
e_mu_2 = mu_2[:,2].detach().numpy()
f_sigma_2 = sigma_2[:,0].detach().numpy()
att_sigma_2 = sigma_2[:,1].detach().numpy()
e_sigma_2 = sigma_2[:,2].detach().numpy()
mean_f_2_vae = f_mu_2 /c
mean_att_2_vae = att_mu_2
mean_e_2_vae = np.exp(e_mu_2 + e_sigma_2**2/2)/c
std_f_2_vae = f_sigma_2/c
std_att_2_vae = att_sigma_2
std_e_2_vae = np.sqrt((np.exp(e_sigma_2**2)-1)*np.exp(2*e_mu_2 + e_sigma_2**2))/c


######################## AVB

ftiss_truth = pd.read_csv("M0_ftiss.csv")/(c*scale)
delttiss_truth = pd.read_csv("delttiss.csv")
ftiss_truth = ftiss_truth.to_numpy()[:, 1:]
ftiss = ftiss_truth.reshape(k, 1, -1)
delttiss_truth = delttiss_truth.to_numpy()[:, 1:]
delttiss = delttiss_truth.reshape(k, 1, -1)
#e_std = (pd.read_csv("e_std.csv").to_numpy()[:, 1:]).reshape(1000, 1, -1)/(c*scale)


outdir_inf = "avb_output/avb_km_snr_10000000"
outdir_10 = "avb_output/avb_km_snr_100"
outdir_5 = "avb_output/avb_km_snr_50"
outdir_2 = "avb_output/avb_km_snr_25"


n = k
factor = 15

mean_f_inf_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_inf).get_data(),(n,))/factor
mean_att_inf_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_inf).get_data(),(n,))
mean_e_inf_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_inf).get_data(),(n,)))/factor
std_f_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_inf).get_data(),(n,)))/factor
std_att_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_inf).get_data(),(n,)))
std_e_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_inf).get_data(),(n,)))/factor


mean_f_10_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_10).get_data(),(n,))/factor
mean_att_10_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_10).get_data(),(n,))
mean_e_10_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_10).get_data(),(n,)))/factor
std_f_10_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_10).get_data(),(n,)))/factor
std_att_10_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_10).get_data(),(n,)))
std_e_10_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_10).get_data(),(n,)))/factor


mean_f_5_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_5).get_data(),(n,))/factor
mean_att_5_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_5).get_data(),(n,))
mean_e_5_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_5).get_data(),(n,)))/factor
std_f_5_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_5).get_data(),(n,)))/factor
std_att_5_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_5).get_data(),(n,)))
std_e_5_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_5).get_data(),(n,)))/factor


mean_f_2_avb = np.reshape(nib.load("%s/mean_ftiss_native.nii.gz" % outdir_2).get_data(),(n,))/factor
mean_att_2_avb = np.reshape(nib.load("%s/mean_delttiss_native.nii.gz" % outdir_2).get_data(),(n,))
mean_e_2_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_2).get_data(),(n,)))/factor
std_f_2_avb = np.sqrt(np.reshape(nib.load("%s/var_ftiss_native.nii.gz" % outdir_2).get_data(),(n,)))/factor
std_att_2_avb = np.sqrt(np.reshape(nib.load("%s/var_delttiss_native.nii.gz" % outdir_2).get_data(),(n,)))
std_e_2_avb = np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_2).get_data(),(n,)))/factor








######################## MCMC
n = 121

ftiss_truth = pd.read_csv("M0_ftiss.csv")/(c*scale)
delttiss_truth = pd.read_csv("delttiss.csv")
ftiss_truth = ftiss_truth.to_numpy()[:, 1:]
ftiss = ftiss_truth.reshape(10000, 1, -1)
delttiss_truth = delttiss_truth.to_numpy()[:, 1:]
delttiss = delttiss_truth.reshape(10000, 1, -1)

nsamp = 10000
factor = 15

f_est_inf = np.array(pd.read_csv('MCMC_output/est_ftiss_inf.csv'))[0:nsamp,1:]/factor
att_est_inf = np.array(pd.read_csv('MCMC_output/est_delt_inf.csv'))[0:nsamp,1:]
e_est_inf = np.array(pd.read_csv('MCMC_output/est_e_inf.csv'))[0:nsamp,1:]/factor
f_est_10 = np.array(pd.read_csv('MCMC_output/est_ftiss_10.csv'))[0:nsamp,1:]/factor
att_est_10 = np.array(pd.read_csv('MCMC_output/est_delt_10.csv'))[0:nsamp,1:]
e_est_10 = np.array(pd.read_csv('MCMC_output/est_e_10.csv'))[0:nsamp,1:]/factor
f_est_5 = np.array(pd.read_csv('MCMC_output/est_ftiss_5.csv'))[0:nsamp,1:]/factor
att_est_5 = np.array(pd.read_csv('MCMC_output/est_delt_5.csv'))[0:nsamp,1:]
e_est_5 = np.array(pd.read_csv('MCMC_output/est_e_5.csv'))[0:nsamp,1:]/factor
f_est_2 = np.array(pd.read_csv('MCMC_output/est_ftiss_2.csv'))[0:nsamp,1:]/factor
att_est_2 = np.array(pd.read_csv('MCMC_output/est_delt_2.csv'))[0:nsamp,1:]
e_est_2 = np.array(pd.read_csv('MCMC_output/est_e_2.csv'))[0:nsamp,1:]/factor



mean_f_inf_mcmc = np.mean(f_est_inf, axis = 0).reshape((n,))
mean_att_inf_mcmc = np.mean(att_est_inf, axis = 0).reshape((n,))
mean_e_inf_mcmc = np.mean(e_est_inf, axis = 0).reshape((n,))
std_e_inf_mcmc = np.quantile(e_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_inf, 0.025, axis=0).reshape((n,))
std_f_inf_mcmc = np.quantile(att_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_inf, 0.025, axis=0).reshape((n,))
std_att_inf_mcmc = np.quantile(f_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_inf, 0.025, axis=0).reshape((n,))

mean_f_10_mcmc = np.mean(f_est_10, axis = 0).reshape((n,))
mean_att_10_mcmc = np.mean(att_est_10, axis = 0).reshape((n,))
mean_e_10_mcmc = np.mean(e_est_10, axis = 0).reshape((n,))
std_e_10_mcmc = np.quantile(e_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_10, 0.025, axis=0).reshape((n,))
std_f_10_mcmc = np.quantile(att_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_10, 0.025, axis=0).reshape((n,))
std_att_10_mcmc = np.quantile(f_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_10, 0.025, axis=0).reshape((n,))

mean_f_5_mcmc = np.mean(f_est_5, axis = 0).reshape((n,))
mean_att_5_mcmc = np.mean(att_est_5, axis = 0).reshape((n,))
mean_e_5_mcmc = np.mean(e_est_5, axis = 0).reshape((n,))
std_e_5_mcmc = np.quantile(e_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_5, 0.025, axis=0).reshape((n,))
std_f_5_mcmc = np.quantile(att_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_5, 0.025, axis=0).reshape((n,))
std_att_5_mcmc = np.quantile(f_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_5, 0.025, axis=0).reshape((n,))

mean_f_2_mcmc = np.mean(f_est_2, axis = 0).reshape((n,))
mean_att_2_mcmc = np.mean(att_est_2, axis = 0).reshape((n,))
mean_e_2_mcmc = np.mean(e_est_2, axis = 0).reshape((n,))
std_e_2_mcmc = np.quantile(e_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_2, 0.025, axis=0).reshape((n,))
std_f_2_mcmc = np.quantile(att_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(att_est_2, 0.025, axis=0).reshape((n,))
std_att_2_mcmc = np.quantile(f_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(f_est_2, 0.025, axis=0).reshape((n,))




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
axes[0,0].set_title('VAE-like, SNR = INF \n General KM',fontsize=8)
axes[0,1].set_title('VAE-like, SNR = 10 \n General KM',fontsize=8)
axes[0,2].set_title('VAE-like, SNR = 5 \n General KM',fontsize=8)
axes[0,3].set_title('VAE-like, SNR = 2 \n General KM',fontsize=8)
im0_avb = axes[1,0].imshow(hmp_avb_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_avb = axes[1,1].imshow(hmp_avb_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_avb = axes[1,2].imshow(hmp_avb_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_avb = axes[1,3].imshow(hmp_avb_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[1,0].set_title('aVB, SNR = INF \n General KM',fontsize=8)
axes[1,1].set_title('aVB, SNR = 10 \n General KM',fontsize=8)
axes[1,2].set_title('aVB, SNR = 5 \n General KM',fontsize=8)
axes[1,3].set_title('aVB, SNR = 2 \n General KM',fontsize=8)
im0_mcmc = axes[2,0].imshow(hmp_mcmc_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_mcmc = axes[2,1].imshow(hmp_mcmc_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_mcmc = axes[2,2].imshow(hmp_mcmc_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_mcmc = axes[2,3].imshow(hmp_mcmc_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[2,0].set_title('MCMC, SNR = INF \n General KM',fontsize=8)
axes[2,1].set_title('MCMC, SNR = 10 \n General KM',fontsize=8)
axes[2,2].set_title('MCMC, SNR = 5 \n General KM',fontsize=8)
axes[2,3].set_title('MCMC, SNR = 2 \n General KM',fontsize=8)
fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.82, 0.32, 0.01, 0.40])
#cbar_ax = fig.add_axes([0.82, 0.1, 0.02, 0.6])
#fig.colorbar(im3_vae, cax=cbar_ax)
fig.tight_layout(pad=0.1)
plt.savefig('density_km.png')
#plt.show()



