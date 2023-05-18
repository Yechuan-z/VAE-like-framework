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
from sklearn.metrics import r2_score
import os.path


import sys
import math
import random
import matplotlib.pyplot as plt
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
k = 1000
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
e_std = (pd.read_csv("e_std.csv").to_numpy()[:, 1:]).reshape(1000, 1, -1)/(c*scale)


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
n = 1000

ftiss_truth = pd.read_csv("M0_ftiss.csv")/(c*scale)
delttiss_truth = pd.read_csv("delttiss.csv")
ftiss_truth = ftiss_truth.to_numpy()[:, 1:]
ftiss = ftiss_truth.reshape(1000, 1, -1)
delttiss_truth = delttiss_truth.to_numpy()[:, 1:]
delttiss = delttiss_truth.reshape(1000, 1, -1)

nsamp = 500
factor = 15

mean_f_inf_mcmc = np.array(pd.read_csv('MCMC_output/mean_f_inf_mcmc.csv'))[0:nsamp,1:]
mean_att_inf_mcmc = np.array(pd.read_csv('MCMC_output/mean_att_inf_mcmc.csv'))[0:nsamp,1:]
mean_e_inf_mcmc = np.array(pd.read_csv('MCMC_output/mean_e_inf_mcmc.csv'))[0:nsamp,1:]
std_e_inf_mcmc = np.array(pd.read_csv('MCMC_output/std_f_inf_mcmc.csv'))[0:nsamp,1:]
std_f_inf_mcmc = np.array(pd.read_csv('MCMC_output/std_att_inf_mcmc.csv'))[0:nsamp,1:]
std_att_inf_mcmc = np.array(pd.read_csv('MCMC_output/std_e_inf_mcmc.csv'))[0:nsamp,1:]

mean_f_10_mcmc = np.array(pd.read_csv('MCMC_output/mean_f_10_mcmc.csv'))[0:nsamp,1:]
mean_att_10_mcmc = np.array(pd.read_csv('MCMC_output/mean_att_10_mcmc.csv'))[0:nsamp,1:]
mean_e_10_mcmc = np.array(pd.read_csv('MCMC_output/mean_e_10_mcmc.csv'))[0:nsamp,1:]
std_e_10_mcmc = np.array(pd.read_csv('MCMC_output/std_f_10_mcmc.csv'))[0:nsamp,1:]
std_f_10_mcmc = np.array(pd.read_csv('MCMC_output/std_att_10_mcmc.csv'))[0:nsamp,1:]
std_att_10_mcmc = np.array(pd.read_csv('MCMC_output/std_e_10_mcmc.csv'))[0:nsamp,1:]

mean_f_5_mcmc = np.array(pd.read_csv('MCMC_output/mean_f_5_mcmc.csv'))[0:nsamp,1:]
mean_att_5_mcmc = np.array(pd.read_csv('MCMC_output/mean_att_5_mcmc.csv'))[0:nsamp,1:]
mean_e_5_mcmc = np.array(pd.read_csv('MCMC_output/mean_e_5_mcmc.csv'))[0:nsamp,1:]
std_e_5_mcmc = np.array(pd.read_csv('MCMC_output/std_f_5_mcmc.csv'))[0:nsamp,1:]
std_f_5_mcmc = np.array(pd.read_csv('MCMC_output/std_att_5_mcmc.csv'))[0:nsamp,1:]
std_att_5_mcmc = np.array(pd.read_csv('MCMC_output/std_e_5_mcmc.csv'))[0:nsamp,1:]

mean_f_2_mcmc = np.array(pd.read_csv('MCMC_output/mean_f_2_mcmc.csv'))[0:nsamp,1:]
mean_att_2_mcmc = np.array(pd.read_csv('MCMC_output/mean_att_2_mcmc.csv'))[0:nsamp,1:]
mean_e_2_mcmc = np.array(pd.read_csv('MCMC_output/mean_e_2_mcmc.csv'))[0:nsamp,1:]
std_e_2_mcmc = np.array(pd.read_csv('MCMC_output/std_f_2_mcmc.csv'))[0:nsamp,1:]
std_f_2_mcmc = np.array(pd.read_csv('MCMC_output/std_att_2_mcmc.csv'))[0:nsamp,1:]
std_att_2_mcmc = np.array(pd.read_csv('MCMC_output/std_e_2_mcmc.csv'))[0:nsamp,1:]





######################## Compare perfusion
########### 1) Mean value

diff_f_inf_vae = np.reshape(mean_f_inf_vae.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_inf_avb = np.reshape(mean_f_inf_avb.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_inf_mcmc = np.reshape(mean_f_inf_mcmc.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_10_vae = np.reshape(mean_f_10_vae.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_10_avb = np.reshape(mean_f_10_avb.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_10_mcmc = np.reshape(mean_f_10_mcmc.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_5_vae = np.reshape(mean_f_5_vae.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_5_avb = np.reshape(mean_f_5_avb.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_5_mcmc = np.reshape(mean_f_5_mcmc.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_2_vae = np.reshape(mean_f_2_vae.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_2_avb = np.reshape(mean_f_2_avb.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))
diff_f_2_mcmc = np.reshape(mean_f_2_mcmc.reshape((k,)) - ftiss.reshape((k,)),(ftiss.shape[0],))

data_vae = [diff_f_inf_vae.tolist(), diff_f_10_vae.tolist(), diff_f_5_vae.tolist(), diff_f_2_vae.tolist()]
data_avb = [diff_f_inf_avb.tolist(), diff_f_10_avb.tolist(), diff_f_5_avb.tolist(), diff_f_2_avb.tolist()]
data_mcmc = [diff_f_inf_mcmc.tolist(), diff_f_10_mcmc.tolist(), diff_f_5_mcmc.tolist(), diff_f_2_mcmc.tolist()]
ticks = ['SNR = INF', 'SNR = 10','SNR = 5','SNR = 2.5']
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
fig = plt.gcf()
fig.set_size_inches(10, 6)
#figure(figsize=(18, 6), dpi=80)
bpl = plt.boxplot(data_vae, positions=np.array(range(len(data_vae)))*2.0-0.45, sym='', widths=0.35)
bpm = plt.boxplot(data_avb, positions=np.array(range(len(data_avb)))*2.0, sym='', widths=0.35)
bpr = plt.boxplot(data_mcmc, positions=np.array(range(len(data_mcmc)))*2.0+0.45, sym='', widths=0.35)
set_box_color(bpl, 'b') 
set_box_color(bpm, 'g')
set_box_color(bpr, 'r')

plt.plot([], c='b', label='VAE')
plt.plot([], c='g', label='aVB')
plt.plot([], c='r', label='MCMC')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-1, 1)
plt.title('Error in Perfusion, General KM')
plt.ylabel('Perfusion (ml/g/min)')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('mean_f_km.png')


#2) 95% CI
var_f_inf_vae = np.reshape(std_f_inf_vae*(1.96*2),(ftiss.shape[0],))
var_f_inf_avb = np.reshape(std_f_inf_avb*(1.96*2),(ftiss.shape[0],))
var_f_inf_mcmc = np.reshape(std_f_inf_mcmc,(ftiss.shape[0],))
var_f_10_vae = np.reshape(std_f_10_vae*(1.96*2),(ftiss.shape[0],))
var_f_10_avb = np.reshape(std_f_10_avb*(1.96*2),(ftiss.shape[0],))
var_f_10_mcmc = np.reshape(std_f_10_mcmc,(ftiss.shape[0],))
var_f_5_vae = np.reshape(std_f_5_vae*(1.96*2),(ftiss.shape[0],))
var_f_5_avb = np.reshape(std_f_5_avb*(1.96*2),(ftiss.shape[0],))
var_f_5_mcmc = np.reshape(std_f_5_mcmc,(ftiss.shape[0],))
var_f_2_vae = np.reshape(std_f_2_vae*(1.96*2),(ftiss.shape[0],))
var_f_2_avb = np.reshape(std_f_2_avb*(1.96*2),(ftiss.shape[0],))
var_f_2_mcmc = np.reshape(std_f_2_mcmc,(ftiss.shape[0],))

data_vae = [var_f_inf_vae.tolist(), var_f_10_vae.tolist(), var_f_5_vae.tolist(), var_f_2_vae.tolist()]
data_avb = [var_f_inf_avb.tolist(), var_f_10_avb.tolist(), var_f_5_avb.tolist(), var_f_2_avb.tolist()]
data_mcmc = [var_f_inf_mcmc.tolist(), var_f_10_mcmc.tolist(), var_f_5_mcmc.tolist(), var_f_2_mcmc.tolist()]
ticks = ['SNR = INF', 'SNR = 10','SNR = 5','SNR = 2.5']
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
fig = plt.gcf()
fig.set_size_inches(10, 6)
#figure(figsize=(18, 6), dpi=80)
bpl = plt.boxplot(data_vae, positions=np.array(range(len(data_vae)))*2.0-0.45, sym='', widths=0.35)
bpm = plt.boxplot(data_avb, positions=np.array(range(len(data_avb)))*2.0, sym='', widths=0.35)
bpr = plt.boxplot(data_mcmc, positions=np.array(range(len(data_mcmc)))*2.0+0.45, sym='', widths=0.35)
set_box_color(bpl, 'b') 
set_box_color(bpm, 'g')
set_box_color(bpr, 'r')

plt.plot([], c='b', label='VAE')
plt.plot([], c='g', label='aVB')
plt.plot([], c='r', label='MCMC')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 3)
plt.title('95% CI of Perfusion, General KM')
plt.ylabel('95% CI')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('var_f_km.png')


######################## Compare ATT
########### 1) Mean value

diff_att_inf_vae = np.reshape(mean_att_inf_vae.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_inf_avb = np.reshape(mean_att_inf_avb.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_inf_mcmc = np.reshape(mean_att_inf_mcmc.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_10_vae = np.reshape(mean_att_10_vae.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_10_avb = np.reshape(mean_att_10_avb.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_10_mcmc = np.reshape(mean_att_10_mcmc.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_5_vae = np.reshape(mean_att_5_vae.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_5_avb = np.reshape(mean_att_5_avb.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_5_mcmc = np.reshape(mean_att_5_mcmc.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_2_vae = np.reshape(mean_att_2_vae.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_2_avb = np.reshape(mean_att_2_avb.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))
diff_att_2_mcmc = np.reshape(mean_att_2_mcmc.reshape((k,)) - delttiss.reshape((k,)),(delttiss.shape[0],))

data_vae = [diff_att_inf_vae.tolist(), diff_att_10_vae.tolist(), diff_att_5_vae.tolist(), diff_att_2_vae.tolist()]
data_avb = [diff_att_inf_avb.tolist(), diff_att_10_avb.tolist(), diff_att_5_avb.tolist(), diff_att_2_avb.tolist()]
data_mcmc = [diff_att_inf_mcmc.tolist(), diff_att_10_mcmc.tolist(), diff_att_5_mcmc.tolist(), diff_att_2_mcmc.tolist()]
ticks = ['SNR = INF', 'SNR = 10','SNR = 5','SNR = 2.5']
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
fig = plt.gcf()
fig.set_size_inches(10, 6)
#figure(figsize=(18, 6), dpi=80)
bpl = plt.boxplot(data_vae, positions=np.array(range(len(data_vae)))*2.0-0.45, sym='', widths=0.35)
bpm = plt.boxplot(data_avb, positions=np.array(range(len(data_avb)))*2.0, sym='', widths=0.35)
bpr = plt.boxplot(data_mcmc, positions=np.array(range(len(data_mcmc)))*2.0+0.45, sym='', widths=0.35)
set_box_color(bpl, 'b') 
set_box_color(bpm, 'g')
set_box_color(bpr, 'r')

plt.plot([], c='b', label='VAE')
plt.plot([], c='g', label='aVB')
plt.plot([], c='r', label='MCMC')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-1, 1)
plt.title('Error in ATT, General KM')
plt.ylabel('ATT (s)')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('mean_att_km.png')


#2) 95% CI
var_att_inf_vae = np.reshape(std_att_inf_vae*(1.96*2),(delttiss.shape[0],))
var_att_inf_avb = np.reshape(std_att_inf_avb*(1.96*2),(delttiss.shape[0],))
var_att_inf_mcmc = np.reshape(std_att_inf_mcmc,(delttiss.shape[0],))
var_att_10_vae = np.reshape(std_att_10_vae*(1.96*2),(delttiss.shape[0],))
var_att_10_avb = np.reshape(std_att_10_avb*(1.96*2),(delttiss.shape[0],))
var_att_10_mcmc = np.reshape(std_att_10_mcmc,(delttiss.shape[0],))
var_att_5_vae = np.reshape(std_att_5_vae*(1.96*2),(delttiss.shape[0],))
var_att_5_avb = np.reshape(std_att_5_avb*(1.96*2),(delttiss.shape[0],))
var_att_5_mcmc = np.reshape(std_att_5_mcmc,(delttiss.shape[0],))
var_att_2_vae = np.reshape(std_att_2_vae*(1.96*2),(delttiss.shape[0],))
var_att_2_avb = np.reshape(std_att_2_avb*(1.96*2),(delttiss.shape[0],))
var_att_2_mcmc = np.reshape(std_att_2_mcmc,(delttiss.shape[0],))

data_vae = [var_att_inf_vae.tolist(), var_att_10_vae.tolist(), var_att_5_vae.tolist(), var_att_2_vae.tolist()]
data_avb = [var_att_inf_avb.tolist(), var_att_10_avb.tolist(), var_att_5_avb.tolist(), var_att_2_avb.tolist()]
data_mcmc = [var_att_inf_mcmc.tolist(), var_att_10_mcmc.tolist(), var_att_5_mcmc.tolist(), var_att_2_mcmc.tolist()]
ticks = ['SNR = INF', 'SNR = 10','SNR = 5','SNR = 2.5']
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
fig = plt.gcf()
fig.set_size_inches(10, 6)
#figure(figsize=(18, 6), dpi=80)
bpl = plt.boxplot(data_vae, positions=np.array(range(len(data_vae)))*2.0-0.45, sym='', widths=0.35)
bpm = plt.boxplot(data_avb, positions=np.array(range(len(data_avb)))*2.0, sym='', widths=0.35)
bpr = plt.boxplot(data_mcmc, positions=np.array(range(len(data_mcmc)))*2.0+0.45, sym='', widths=0.35)
set_box_color(bpl, 'b') 
set_box_color(bpm, 'g')
set_box_color(bpr, 'r')

plt.plot([], c='b', label='VAE')
plt.plot([], c='g', label='aVB')
plt.plot([], c='r', label='MCMC')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 3)
plt.title('95% CI of ATT, General KM')
plt.ylabel('95% CI')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('var_att_km.png')




######################## Compare Noise 
########### 1) Mean 
diff_e_inf_vae = np.reshape(mean_e_inf_vae.reshape((k,)) - 0,(e_std.shape[0],))
diff_e_inf_avb = np.reshape(mean_e_inf_avb.reshape((k,)) - 0,(e_std.shape[0],))
diff_e_inf_mcmc = np.reshape(mean_e_inf_mcmc.reshape((k,)) - 0,(e_std.shape[0],))
diff_e_10_vae = np.reshape(mean_e_10_vae.reshape((k,)) - 0.6/10,(e_std.shape[0],))
diff_e_10_avb = np.reshape(mean_e_10_avb.reshape((k,)) - 0.6/10,(e_std.shape[0],))
diff_e_10_mcmc = np.reshape(mean_e_10_mcmc.reshape((k,)) - 0.6/10,(e_std.shape[0],))
diff_e_5_vae = np.reshape(mean_e_5_vae.reshape((k,)) - 0.6/5,(e_std.shape[0],))
diff_e_5_avb = np.reshape(mean_e_5_avb.reshape((k,)) - 0.6/5,(e_std.shape[0],))
diff_e_5_mcmc = np.reshape(mean_e_5_mcmc.reshape((k,)) - 0.6/5,(e_std.shape[0],))
diff_e_2_vae = np.reshape(mean_e_2_vae.reshape((k,)) - 0.6/2.5,(e_std.shape[0],))
diff_e_2_avb = np.reshape(mean_e_2_avb.reshape((k,)) - 0.6/2.5,(e_std.shape[0],))
diff_e_2_mcmc = np.reshape(mean_e_2_mcmc.reshape((k,)) - 0.6/2.5,(e_std.shape[0],))

data_vae = [diff_e_inf_vae.tolist(), diff_e_10_vae.tolist(), diff_e_5_vae.tolist(), diff_e_2_vae.tolist()]
data_avb = [diff_e_inf_avb.tolist(), diff_e_10_avb.tolist(), diff_e_5_avb.tolist(), diff_e_2_avb.tolist()]
data_mcmc = [diff_e_inf_mcmc.tolist(), diff_e_10_mcmc.tolist(), diff_e_5_mcmc.tolist(), diff_e_2_mcmc.tolist()]
ticks = ['SNR = INF', 'SNR = 10','SNR = 5','SNR = 2.5']
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
fig = plt.gcf()
fig.set_size_inches(10, 6)
#figure(figsize=(18, 6), dpi=80)
bpl = plt.boxplot(data_vae, positions=np.array(range(len(data_vae)))*2.0-0.45, sym='', widths=0.35)
bpm = plt.boxplot(data_avb, positions=np.array(range(len(data_avb)))*2.0, sym='', widths=0.35)
bpr = plt.boxplot(data_mcmc, positions=np.array(range(len(data_mcmc)))*2.0+0.45, sym='', widths=0.35)
set_box_color(bpl, 'b') 
set_box_color(bpm, 'g')
set_box_color(bpr, 'r')

plt.plot([], c='b', label='VAE')
plt.plot([], c='g', label='aVB')
plt.plot([], c='r', label='MCMC')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-1, 1)
plt.title('Error in Noise Parameter, General KM')
plt.ylabel('Error')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('mean_e_km.png')

