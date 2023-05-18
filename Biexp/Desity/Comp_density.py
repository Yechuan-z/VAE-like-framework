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
import matplotlib.pyplot as plt
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
    def __init__(self, input_dim= 8, enc_out_dim=100, latent_dim=3):
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
        )
        self.decoder = self.signal
        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_log_std = nn.Linear(enc_out_dim, latent_dim)
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0]))
    def signal(self,t, A, B, a, b):
        signal = A*torch.exp(-a*t) + B*torch.exp(-b*t)
        return signal



########################## Load VAE
path = 'Biexp/VAE_like_biexp.pth'
vae = torch.load(path)


########################## Load Data
m = 0
k = 121
n = k+m

a = np.reshape(np.array(pd.read_csv('little_a.csv'))[m:n,1:],(k,1))
b = np.array(pd.read_csv('little_b.csv'))[m:n,1:]
e_log = np.array(pd.read_csv('e_log.csv'))[m:n,1:]
e_std = np.exp(e_log)
A = np.array(pd.read_csv('a.csv'))[m:n,1:]
B = np.array(pd.read_csv('b.csv'))[m:n,1:]
signal = np.array(pd.read_csv('signal.csv'))[m:n,1:]
signal_noise_inf = np.array(pd.read_csv('signal_noise_snr_10000000.csv'))[m:n,1:]
signal_noise_10 = np.array(pd.read_csv('signal_noise_snr_100.csv'))[m:n,1:]
signal_noise_5 = np.array(pd.read_csv('signal_noise_snr_50.csv'))[m:n,1:]
signal_noise_2 = np.array(pd.read_csv('signal_noise_snr_25.csv'))[m:n,1:]



x_data_inf = np.zeros((k,5+2*signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_inf[:,0:sig_dim] = np.reshape(signal_noise_inf,(k,signal_noise_inf.shape[1]))
x_data_inf[:,sig_dim] = np.reshape(A,(k,))
x_data_inf[:,sig_dim+1] = np.reshape(b,(k,))
x_data_inf[:,sig_dim+2] = np.reshape(a,(k,))
x_data_inf[:,sig_dim+3] = np.reshape(B,(k,))
x_data_inf[:,sig_dim+4] = np.reshape(e_std,(k,))
x_data_inf[:,sig_dim+5:5+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_inf= torch.from_numpy(x_data_inf)




x_data_10 = np.zeros((k,5+2*signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_10[:,0:sig_dim] = np.reshape(signal_noise_10,(k,signal_noise_10.shape[1]))
x_data_10[:,sig_dim] = np.reshape(A,(k,))
x_data_10[:,sig_dim+1] = np.reshape(b,(k,))
x_data_10[:,sig_dim+2] = np.reshape(a,(k,))
x_data_10[:,sig_dim+3] = np.reshape(B,(k,))
x_data_10[:,sig_dim+4] = np.reshape(e_std,(k,))
x_data_10[:,sig_dim+5:5+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_10= torch.from_numpy(x_data_10) 


x_data_5 = np.zeros((k,5+2*signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_5[:,0:sig_dim] = np.reshape(signal_noise_5,(k,signal_noise_5.shape[1]))
x_data_5[:,sig_dim] = np.reshape(A,(k,))
x_data_5[:,sig_dim+1] = np.reshape(b,(k,))
x_data_5[:,sig_dim+2] = np.reshape(a,(k,))
x_data_5[:,sig_dim+3] = np.reshape(B,(k,))
x_data_5[:,sig_dim+4] = np.reshape(e_std,(k,))
x_data_5[:,sig_dim+5:5+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_5= torch.from_numpy(x_data_5)



x_data_2 = np.zeros((k,5+2*signal.shape[1]),np.float32)
sig_dim = signal.shape[1]
x_data_2[:,0:sig_dim] = np.reshape(signal_noise_2,(k,signal_noise_2.shape[1]))
x_data_2[:,sig_dim] = np.reshape(A,(k,))
x_data_2[:,sig_dim+1] = np.reshape(b,(k,))
x_data_2[:,sig_dim+2] = np.reshape(a,(k,))
x_data_2[:,sig_dim+3] = np.reshape(B,(k,))
x_data_2[:,sig_dim+4] = np.reshape(e_std,(k,))
x_data_2[:,sig_dim+5:5+2*sig_dim] = np.reshape(signal,(k,signal.shape[1]))
x_data_2= torch.from_numpy(x_data_2)










######################################################## Comparison

######################## Run VAE
pred_encoder_inf = vae.encoder(x_data_inf[:,0:8])
mu_inf = vae.fc_mu(pred_encoder_inf)
sigma_inf = torch.exp(vae.fc_log_std(pred_encoder_inf))
a_mu_inf = mu_inf[:,0].detach().numpy()
b_mu_inf = mu_inf[:,1].detach().numpy()
e_mu_inf = mu_inf[:,2].detach().numpy()
a_sigma_inf = sigma_inf[:,0].detach().numpy()
b_sigma_inf = sigma_inf[:,1].detach().numpy()
e_sigma_inf = sigma_inf[:,2].detach().numpy()
mean_a_inf_vae = a_mu_inf 
mean_b_inf_vae = b_mu_inf 
mean_e_inf_vae = np.exp(e_mu_inf + e_sigma_inf**2/2)
std_a_inf_vae = a_sigma_inf
std_b_inf_vae = b_sigma_inf
std_e_inf_vae = np.sqrt((np.exp(e_sigma_inf**2)-1)*np.exp(2*e_mu_inf + e_sigma_inf**2))


pred_encoder_10 = vae.encoder(x_data_10[:,0:8])
mu_10 = vae.fc_mu(pred_encoder_10)
sigma_10 = torch.exp(vae.fc_log_std(pred_encoder_10))
a_mu_10 = mu_10[:,0].detach().numpy()
b_mu_10 = mu_10[:,1].detach().numpy()
e_mu_10 = mu_10[:,2].detach().numpy()
a_sigma_10 = sigma_10[:,0].detach().numpy()
b_sigma_10 = sigma_10[:,1].detach().numpy()
e_sigma_10 = sigma_10[:,2].detach().numpy()
mean_a_10_vae = a_mu_10 
mean_b_10_vae = b_mu_10 
mean_e_10_vae = np.exp(e_mu_10 + e_sigma_10**2/2)
std_a_10_vae = a_sigma_10
std_b_10_vae = b_sigma_10
std_e_10_vae = np.sqrt((np.exp(e_sigma_10**2)-1)*np.exp(2*e_mu_10 + e_sigma_10**2))


pred_encoder_5 = vae.encoder(x_data_5[:,0:8])
mu_5 = vae.fc_mu(pred_encoder_5)
sigma_5 = torch.exp(vae.fc_log_std(pred_encoder_5))
a_mu_5 = mu_5[:,0].detach().numpy()
b_mu_5 = mu_5[:,1].detach().numpy()
e_mu_5 = mu_5[:,2].detach().numpy()
a_sigma_5 = sigma_5[:,0].detach().numpy()
b_sigma_5 = sigma_5[:,1].detach().numpy()
e_sigma_5 = sigma_5[:,2].detach().numpy()
mean_a_5_vae = a_mu_5 
mean_b_5_vae = b_mu_5 
mean_e_5_vae = np.exp(e_mu_5 + e_sigma_5**2/2)
std_a_5_vae = a_sigma_5
std_b_5_vae = b_sigma_5
std_e_5_vae = np.sqrt((np.exp(e_sigma_5**2)-1)*np.exp(2*e_mu_5 + e_sigma_5**2))


pred_encoder_2 = vae.encoder(x_data_2[:,0:8])
mu_2 = vae.fc_mu(pred_encoder_2)
sigma_2 = torch.exp(vae.fc_log_std(pred_encoder_2))
a_mu_2 = mu_2[:,0].detach().numpy()
b_mu_2 = mu_2[:,1].detach().numpy()
e_mu_2 = mu_2[:,2].detach().numpy()
a_sigma_2 = sigma_2[:,0].detach().numpy()
b_sigma_2 = sigma_2[:,1].detach().numpy()
e_sigma_2 = sigma_2[:,2].detach().numpy()
mean_a_2_vae = a_mu_2
mean_b_2_vae = b_mu_2 
mean_e_2_vae = np.exp(e_mu_2 + e_sigma_2**2/2)
std_a_2_vae = a_sigma_2
std_b_2_vae = b_sigma_2
std_e_2_vae = np.sqrt((np.exp(e_sigma_2**2)-1)*np.exp(2*e_mu_2 + e_sigma_2**2))


######################## AVB

n = 121

a_truth = pd.read_csv("little_a.csv")
B_truth = pd.read_csv("B.csv")
a_truth = a_truth.to_numpy()[:, 1:]
a = a_truth.reshape(n,)
B_truth = B_truth.to_numpy()[:, 1:]
B = B_truth.reshape(n,)


outdir_inf = "avb_output/avb_biexp_snr_10000000"
outdir_10 = "avb_output/avb_biexp_snr_100"
outdir_5 = "avb_output/avb_biexp_snr_50"
outdir_2 = "avb_output/avb_biexp_snr_25"

mean_a_inf_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_inf).get_data(),(n,))
mean_b_inf_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_inf).get_data(),(n,))
mean_e_inf_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_inf).get_data(),(n,)))
std_a_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_inf).get_data(),(n,)))
std_b_inf_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_inf).get_data(),(n,)))
std_e_inf_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_inf).get_data(),(n,)))
cov_inf_avb = np.reshape((nib.load("%s/posterior.nii.gz" % outdir_inf).get_data())[:,:,:,1],(n,))


mean_a_10_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_10).get_data(),(n,))
mean_b_10_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_10).get_data(),(n,))
mean_e_10_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_10).get_data(),(n,)))
std_a_10_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_10).get_data(),(n,)))
std_b_10_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_10).get_data(),(n,)))
std_e_10_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_10).get_data(),(n,)))
cov_10_avb = np.reshape((nib.load("%s/posterior.nii.gz" % outdir_10).get_data())[:,:,:,1],(n,))


mean_a_5_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_5).get_data(),(n,))
mean_b_5_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_5).get_data(),(n,))
mean_e_5_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_5).get_data(),(n,)))
std_a_5_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_5).get_data(),(n,)))
std_b_5_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_5).get_data(),(n,)))
std_e_5_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_5).get_data(),(n,)))
cov_5_avb = np.reshape((nib.load("%s/posterior.nii.gz" % outdir_5).get_data())[:,:,:,1],(n,))


mean_a_2_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_2).get_data(),(n,))
mean_b_2_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_2).get_data(),(n,))
mean_e_2_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_2).get_data(),(n,)))
std_a_2_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_2).get_data(),(n,)))
std_b_2_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_2).get_data(),(n,)))
std_e_2_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_2).get_data(),(n,)))
cov_2_avb = np.reshape((nib.load("%s/posterior.nii.gz" % outdir_2).get_data())[:,:,:,1],(n,))







######################## MCMC
n = 121

a_truth = pd.read_csv("little_a.csv")
B_truth = pd.read_csv("B.csv")
a_truth = a_truth.to_numpy()[:, 1:]
a = a_truth.reshape(n,)
B_truth = B_truth.to_numpy()[:, 1:]
B = B_truth.reshape(n,)
nsamp = 10000

a_est_inf = np.array(pd.read_csv('MCMC_output/est_a_inf.csv'))[0:nsamp,1:]
B_est_inf = np.array(pd.read_csv('MCMC_output/est_B_inf.csv'))[0:nsamp,1:]
e_est_inf = np.array(pd.read_csv('MCMC_output/est_e_inf.csv'))[0:nsamp,1:]
a_est_10 = np.array(pd.read_csv('MCMC_output/est_a_10.csv'))[0:nsamp,1:]
B_est_10 = np.array(pd.read_csv('MCMC_output/est_B_10.csv'))[0:nsamp,1:]
e_est_10 = np.array(pd.read_csv('MCMC_output/est_e_10.csv'))[0:nsamp,1:]
a_est_5 = np.array(pd.read_csv('MCMC_output/est_a_5.csv'))[0:nsamp,1:]
B_est_5 = np.array(pd.read_csv('MCMC_output/est_B_5.csv'))[0:nsamp,1:]
e_est_5 = np.array(pd.read_csv('MCMC_output/est_e_5.csv'))[0:nsamp,1:]
a_est_2 = np.array(pd.read_csv('MCMC_output/est_a_2.csv'))[0:nsamp,1:]
B_est_2 = np.array(pd.read_csv('MCMC_output/est_B_2.csv'))[0:nsamp,1:]
e_est_2 = np.array(pd.read_csv('MCMC_output/est_e_2.csv'))[0:nsamp,1:]


mean_a_inf_mcmc = np.mean(a_est_inf, axis = 0).reshape((n,))
mean_B_inf_mcmc = np.mean(B_est_inf, axis = 0).reshape((n,))
mean_e_inf_mcmc = np.mean(e_est_inf, axis = 0).reshape((n,))
std_e_inf_mcmc = np.quantile(e_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_inf, 0.025, axis=0).reshape((n,))
std_a_inf_mcmc = np.quantile(a_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(a_est_inf, 0.025, axis=0).reshape((n,))
std_B_inf_mcmc = np.quantile(B_est_inf, 0.975, axis=0).reshape((k,)) - np.quantile(B_est_inf, 0.025, axis=0).reshape((n,))

mean_a_10_mcmc = np.mean(a_est_10, axis = 0).reshape((n,))
mean_B_10_mcmc = np.mean(B_est_10, axis = 0).reshape((n,))
mean_e_10_mcmc = np.mean(e_est_10, axis = 0).reshape((n,))
std_e_10_mcmc = np.quantile(e_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_10, 0.025, axis=0).reshape((n,))
std_a_10_mcmc = np.quantile(a_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(a_est_10, 0.025, axis=0).reshape((n,))
std_B_10_mcmc = np.quantile(B_est_10, 0.975, axis=0).reshape((k,)) - np.quantile(B_est_10, 0.025, axis=0).reshape((n,))

mean_a_5_mcmc = np.mean(a_est_5, axis = 0).reshape((n,))
mean_B_5_mcmc = np.mean(B_est_5, axis = 0).reshape((n,))
mean_e_5_mcmc = np.mean(e_est_5, axis = 0).reshape((n,))
std_e_5_mcmc = np.quantile(e_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_5, 0.025, axis=0).reshape((n,))
std_a_5_mcmc = np.quantile(a_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(a_est_5, 0.025, axis=0).reshape((n,))
std_B_5_mcmc = np.quantile(B_est_5, 0.975, axis=0).reshape((k,)) - np.quantile(B_est_5, 0.025, axis=0).reshape((n,))

mean_a_2_mcmc = np.mean(a_est_2, axis = 0).reshape((n,))
mean_B_2_mcmc = np.mean(B_est_2, axis = 0).reshape((n,))
mean_e_2_mcmc = np.mean(e_est_2, axis = 0).reshape((n,))
std_e_2_mcmc = np.quantile(e_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(e_est_2, 0.025, axis=0).reshape((n,))
std_a_2_mcmc = np.quantile(a_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(a_est_2, 0.025, axis=0).reshape((n,))
std_B_2_mcmc = np.quantile(B_est_2, 0.975, axis=0).reshape((k,)) - np.quantile(B_est_2, 0.025, axis=0).reshape((n,))



######## Heatmap 
####### a = 1, B = 1
x = np.linspace(0,3,51)
y = np.linspace(0,3,51)
int_x = x[1]-x[0]
int_y = y[1]-y[0]

###### 1) VAE

chk_x = 60
chk_y = 60

dist_vae_inf = multivariate_normal(mean=[mean_a_inf_vae[chk_x,], mean_b_inf_vae[chk_y,]], cov=[[std_a_inf_vae[chk_x,],0],[0,std_b_inf_vae[chk_y,]]])
hmp_vae_inf = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_inf[i,j] = dist_vae_inf.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_inf.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_inf.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_inf.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_vae_10 = multivariate_normal(mean=[mean_a_10_vae[chk_x,], mean_b_10_vae[chk_y,]], cov=[[std_a_10_vae[chk_x,],0],[0,std_b_10_vae[chk_y,]]])
hmp_vae_10 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_10[i,j] = dist_vae_10.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_10.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_10.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_10.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_vae_5 = multivariate_normal(mean=[mean_a_5_vae[chk_x,], mean_b_5_vae[chk_y,]], cov=[[std_a_5_vae[chk_x,],0],[0,std_b_5_vae[chk_y,]]])
hmp_vae_5 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_5[i,j] = dist_vae_5.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_5.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_5.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_5.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_vae_2 = multivariate_normal(mean=[mean_a_2_vae[chk_x,], mean_b_2_vae[chk_y,]], cov=[[std_a_2_vae[chk_x,],0],[0,std_b_2_vae[chk_y,]]])
hmp_vae_2 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_vae_2[i,j] = dist_vae_2.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_vae_2.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_vae_2.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_vae_2.cdf([x[i,]-int_x/2,y[j,]-int_y/2])





dist_avb_inf = multivariate_normal(mean=[mean_a_inf_avb[chk_x,], mean_b_inf_avb[chk_y,]], cov=[[std_a_inf_avb[chk_x,],cov_inf_avb[chk_x,]],[cov_inf_avb[chk_x,],std_b_inf_avb[chk_y,]]])
hmp_avb_inf = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_inf[i,j] = dist_avb_inf.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_inf.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_inf.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_inf.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_avb_10 = multivariate_normal(mean=[mean_a_10_avb[chk_x,], mean_b_10_avb[chk_y,]], cov=[[std_a_10_avb[chk_x,],cov_10_avb[chk_x,]],[cov_10_avb[chk_x,],std_b_10_avb[chk_y,]]])
hmp_avb_10 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_10[i,j] = dist_avb_10.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_10.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_10.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_10.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_avb_5 = multivariate_normal(mean=[mean_a_5_avb[chk_x,], mean_b_5_avb[chk_y,]], cov=[[std_a_5_avb[chk_x,],cov_5_avb[chk_x,]],[cov_5_avb[chk_x,],std_b_5_avb[chk_y,]]])
hmp_avb_5 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_5[i,j] = dist_avb_5.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_5.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_5.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_5.cdf([x[i,]-int_x/2,y[j,]-int_y/2])

dist_avb_2 = multivariate_normal(mean=[mean_a_2_avb[chk_x,], mean_b_2_avb[chk_y,]], cov=[[std_a_2_avb[chk_x,],cov_2_avb[chk_x,]],[cov_2_avb[chk_x,],std_b_2_avb[chk_y,]]])
hmp_avb_2 = np.zeros((x.shape[0],y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        hmp_avb_2[i,j] = dist_avb_2.cdf([x[i,]+int_x/2,y[j,]+int_y/2]) - dist_avb_2.cdf([x[i,]-int_x/2,y[j,]+ int_y/2]) - dist_avb_2.cdf([x[i,]+int_x/2,y[j,]-int_y/2]) + dist_avb_2.cdf([x[i,]-int_x/2,y[j,]-int_y/2])






dist_a_mcmc_inf = a_est_inf[:,chk_x]
dist_b_mcmc_inf = B_est_inf[:,chk_y]
hmp_mcmc_inf = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_a_mcmc_inf>x[i]-int_x/2,dist_a_mcmc_inf<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_b_mcmc_inf>y[j]-int_y/2,dist_b_mcmc_inf<=y[j]+int_y/2)
        hmp_mcmc_inf[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp

dist_a_mcmc_10 = a_est_10[:,chk_x]
dist_b_mcmc_10 = B_est_10[:,chk_y]
hmp_mcmc_10 = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_a_mcmc_10>x[i]-int_x/2,dist_a_mcmc_10<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_b_mcmc_10>y[j]-int_y/2,dist_b_mcmc_10<=y[j]+int_y/2)
        hmp_mcmc_10[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp


dist_a_mcmc_5 = a_est_5[:,chk_x]
dist_b_mcmc_5 = B_est_5[:,chk_y]
hmp_mcmc_5 = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_a_mcmc_5>x[i]-int_x/2,dist_a_mcmc_5<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_b_mcmc_5>y[j]-int_y/2,dist_b_mcmc_5<=y[j]+int_y/2)
        hmp_mcmc_5[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp

dist_a_mcmc_2 = a_est_2[:,chk_x]
dist_b_mcmc_2 = B_est_2[:,chk_y]
hmp_mcmc_2 = np.zeros((x.shape[0],y.shape[0]))
int_x = x[1]-x[0]
int_y = y[1]-y[0]
for i in range(x.shape[0]):
    x_logic = np.logical_and(dist_a_mcmc_2>x[i]-int_x/2,dist_a_mcmc_2<=x[i]+int_x/2)
    for j in range(y.shape[0]):
        y_logic = np.logical_and(dist_b_mcmc_2>y[j]-int_y/2,dist_b_mcmc_2<=y[j]+int_y/2)
        hmp_mcmc_2[i,j] = np.sum(np.logical_and(x_logic,y_logic))/nsamp







fig, axes = plt.subplots(nrows=3, ncols=4)
fig.set_size_inches(9, 8)
fig.suptitle('Bi-exponential Model, Estimated Density Plot for a = 1, B = 1', fontsize=14, y = 0.95)
extent = [x.min(), x.max(), y.min(), y.max()]
im0_vae = axes[0,0].imshow(hmp_vae_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_vae = axes[0,1].imshow(hmp_vae_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_vae = axes[0,2].imshow(hmp_vae_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_vae = axes[0,3].imshow(hmp_vae_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[0,0].set_title('VAE-like, SNR = INF ',fontsize=10)
axes[0,1].set_title('VAE-like, SNR = 10 ',fontsize=10)
axes[0,2].set_title('VAE-like, SNR = 5 ',fontsize=10)
axes[0,3].set_title('VAE-like, SNR = 2 ',fontsize=10)
im0_avb = axes[1,0].imshow(hmp_avb_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_avb = axes[1,1].imshow(hmp_avb_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_avb = axes[1,2].imshow(hmp_avb_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_avb = axes[1,3].imshow(hmp_avb_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[1,0].set_title('aVB, SNR = INF ',fontsize=10)
axes[1,1].set_title('aVB, SNR = 10 ',fontsize=10)
axes[1,2].set_title('aVB, SNR = 5 ',fontsize=10)
axes[1,3].set_title('aVB, SNR = 2 ',fontsize=10)
im0_mcmc = axes[2,0].imshow(hmp_mcmc_inf, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im1_mcmc = axes[2,1].imshow(hmp_mcmc_10, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im2_mcmc = axes[2,2].imshow(hmp_mcmc_5, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
im3_mcmc = axes[2,3].imshow(hmp_mcmc_2, vmin=0, vmax=0.005, cmap="YlGnBu", extent = extent)
axes[2,0].set_title('MCMC, SNR = INF ',fontsize=10)
axes[2,1].set_title('MCMC, SNR = 10 ',fontsize=10)
axes[2,2].set_title('MCMC, SNR = 5 ',fontsize=10)
axes[2,3].set_title('MCMC, SNR = 2 ',fontsize=10)
axes[0,0].set_xticklabels(range(4), fontsize=6)
axes[0,1].set_xticklabels(range(4), fontsize=6)
axes[0,2].set_xticklabels(range(4), fontsize=6)
axes[0,3].set_xticklabels(range(4), fontsize=6)
axes[1,0].set_xticklabels(range(4), fontsize=6)
axes[1,1].set_xticklabels(range(4), fontsize=6)
axes[1,2].set_xticklabels(range(4), fontsize=6)
axes[1,3].set_xticklabels(range(4), fontsize=6)
axes[2,0].set_xticklabels(range(4), fontsize=6)
axes[2,1].set_xticklabels(range(4), fontsize=6)
axes[2,2].set_xticklabels(range(4), fontsize=6)
axes[2,3].set_xticklabels(range(4), fontsize=6)
axes[0,0].set_yticklabels(range(4), fontsize=6)
axes[0,1].set_yticklabels(range(4), fontsize=6)
axes[0,2].set_yticklabels(range(4), fontsize=6)
axes[0,3].set_yticklabels(range(4), fontsize=6)
axes[1,0].set_yticklabels(range(4), fontsize=6)
axes[1,1].set_yticklabels(range(4), fontsize=6)
axes[1,2].set_yticklabels(range(4), fontsize=6)
axes[1,3].set_yticklabels(range(4), fontsize=6)
axes[2,0].set_yticklabels(range(4), fontsize=6)
axes[2,1].set_yticklabels(range(4), fontsize=6)
axes[2,2].set_yticklabels(range(4), fontsize=6)
axes[2,3].set_yticklabels(range(4), fontsize=6)
axes[0,0].set_xlabel('Rate a',fontsize=8, loc='left')
axes[0,0].set_ylabel('Amplitude B',fontsize=8, loc='top')
axes[1,0].set_xlabel('Rate a',fontsize=8, loc='left')
axes[1,0].set_ylabel('Amplitude B',fontsize=8, loc='top')
axes[2,0].set_xlabel('Rate a',fontsize=8, loc='left')
axes[2,0].set_ylabel('Amplitude B',fontsize=8, loc='top')
cbar_ax = fig.add_axes([0.94, 0.12, 0.009, 0.75])
cbar = fig.colorbar(im3_vae, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
plt.savefig('density_biexp.png')
#plt.show()


