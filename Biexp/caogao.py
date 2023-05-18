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
from matplotlib.pyplot import figure
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
k = 1000
n = k+m

a = np.array(pd.read_csv('little_a.csv'))[m:n,1:]
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

n = 1000

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


mean_a_10_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_10).get_data(),(n,))
mean_b_10_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_10).get_data(),(n,))
mean_e_10_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_10).get_data(),(n,)))
std_a_10_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_10).get_data(),(n,)))
std_b_10_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_10).get_data(),(n,)))
std_e_10_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_10).get_data(),(n,)))


mean_a_5_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_5).get_data(),(n,))
mean_b_5_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_5).get_data(),(n,))
mean_e_5_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_5).get_data(),(n,)))
std_a_5_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_5).get_data(),(n,)))
std_b_5_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_5).get_data(),(n,)))
std_e_5_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_5).get_data(),(n,)))


mean_a_2_avb = np.reshape(nib.load("%s/mean_rate1_native.nii.gz" % outdir_2).get_data(),(n,))
mean_b_2_avb = np.reshape(nib.load("%s/mean_amp2_native.nii.gz" % outdir_2).get_data(),(n,))
mean_e_2_avb = 1/np.sqrt(np.reshape(nib.load("%s/mean_noise.nii.gz" % outdir_2).get_data(),(n,)))
std_a_2_avb = np.sqrt(np.reshape(nib.load("%s/var_rate1_native.nii.gz" % outdir_2).get_data(),(n,)))
std_b_2_avb = np.sqrt(np.reshape(nib.load("%s/var_amp2_native.nii.gz" % outdir_2).get_data(),(n,)))
std_e_2_avb = 1/np.sqrt(np.reshape(nib.load("%s/var_noise.nii.gz" % outdir_2).get_data(),(n,)))







######################## MCMC
n = 1000

a_truth = pd.read_csv("little_a.csv")
B_truth = pd.read_csv("B.csv")
a_truth = a_truth.to_numpy()[:, 1:]
a = a_truth.reshape(n,)
B_truth = B_truth.to_numpy()[:, 1:]
B = B_truth.reshape(n,)
nsamp = 5000

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



######################## Compare Rate a

#1) Mean 
diff_a_inf_vae = np.reshape(mean_a_inf_vae.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_inf_avb = np.reshape(mean_a_inf_avb.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_inf_mcmc = np.reshape(mean_a_inf_mcmc.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_10_vae = np.reshape(mean_a_10_vae.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_10_avb = np.reshape(mean_a_10_avb.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_10_mcmc = np.reshape(mean_a_10_mcmc.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_5_vae = np.reshape(mean_a_5_vae.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_5_avb = np.reshape(mean_a_5_avb.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_5_mcmc = np.reshape(mean_a_5_mcmc.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_2_vae = np.reshape(mean_a_2_vae.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_2_avb = np.reshape(mean_a_2_avb.reshape((k,)) - a.reshape((k,)),(a.shape[0],))
diff_a_2_mcmc = np.reshape(mean_a_2_mcmc.reshape((k,)) - a.reshape((k,)),(a.shape[0],))

data_vae = [diff_a_inf_vae.tolist(), diff_a_10_vae.tolist(), diff_a_5_vae.tolist(), diff_a_2_vae.tolist()]
data_avb = [diff_a_inf_avb.tolist(), diff_a_10_avb.tolist(), diff_a_5_avb.tolist(), diff_a_2_avb.tolist()]
data_mcmc = [diff_a_inf_mcmc.tolist(), diff_a_10_mcmc.tolist(), diff_a_5_mcmc.tolist(), diff_a_2_mcmc.tolist()]
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
plt.ylim(-2, 2)
plt.title('Error in Rate a, Bi-exponential Model')
plt.ylabel('Error')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('mean_a_biexp.png')



#2) 95% CI
var_a_inf_vae = np.reshape(std_a_inf_vae*(1.96*2),(a.shape[0],))
var_a_inf_avb = np.reshape(std_a_inf_avb*(1.96*2),(a.shape[0],))
var_a_inf_mcmc = std_a_inf_mcmc
var_a_10_vae = np.reshape(std_a_10_vae*(1.96*2),(a.shape[0],))
var_a_10_avb = np.reshape(std_a_10_avb*(1.96*2),(a.shape[0],))
var_a_10_mcmc = std_a_10_mcmc
var_a_5_vae = np.reshape(std_a_5_vae*(1.96*2),(a.shape[0],))
var_a_5_avb = np.reshape(std_a_5_avb*(1.96*2),(a.shape[0],))
var_a_5_mcmc = std_a_5_mcmc
var_a_2_vae = np.reshape(std_a_2_vae*(1.96*2),(a.shape[0],))
var_a_2_avb = np.reshape(std_a_2_avb*(1.96*2),(a.shape[0],))
var_a_2_mcmc = std_a_2_mcmc

data_vae = [var_a_inf_vae.tolist(), var_a_10_vae.tolist(), var_a_5_vae.tolist(), var_a_2_vae.tolist()]
data_avb = [var_a_inf_avb.tolist(), var_a_10_avb.tolist(), var_a_5_avb.tolist(), var_a_2_avb.tolist()]
data_mcmc = [var_a_inf_mcmc.tolist(), var_a_10_mcmc.tolist(), var_a_5_mcmc.tolist(), var_a_2_mcmc.tolist()]
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
plt.ylim(0, 10)
plt.title('95% CI of Rate a, Bi-exponential Model')
plt.ylabel('95% CI')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('var_a_biexp.png')






######################## Compare Amplitude B

#1) Mean 
diff_b_inf_vae = np.reshape(mean_b_inf_vae.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_inf_avb = np.reshape(mean_b_inf_avb.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_inf_mcmc = np.reshape(mean_B_inf_mcmc.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_10_vae = np.reshape(mean_b_10_vae.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_10_avb = np.reshape(mean_b_10_avb.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_10_mcmc = np.reshape(mean_B_10_mcmc.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_5_vae = np.reshape(mean_b_5_vae.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_5_avb = np.reshape(mean_b_5_avb.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_5_mcmc = np.reshape(mean_B_5_mcmc.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_2_vae = np.reshape(mean_b_2_vae.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_2_avb = np.reshape(mean_b_2_avb.reshape((k,)) - B.reshape((k,)),(B.shape[0],))
diff_b_2_mcmc = np.reshape(mean_B_2_mcmc.reshape((k,)) - B.reshape((k,)),(B.shape[0],))

data_vae = [diff_b_inf_vae.tolist(), diff_b_10_vae.tolist(), diff_b_5_vae.tolist(), diff_b_2_vae.tolist()]
data_avb = [diff_b_inf_avb.tolist(), diff_b_10_avb.tolist(), diff_b_5_avb.tolist(), diff_b_2_avb.tolist()]
data_mcmc = [diff_b_inf_mcmc.tolist(), diff_b_10_mcmc.tolist(), diff_b_5_mcmc.tolist(), diff_b_2_mcmc.tolist()]
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
plt.ylim(-2, 2)
plt.title('Error in Amplitude B, Bi-exponential Model')
plt.ylabel('Error')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('mean_B_biexp.png')


#2) CI
var_b_inf_vae = np.reshape(std_b_inf_vae*(1.96*2),(a.shape[0],))
var_b_inf_avb = np.reshape(std_b_inf_avb*(1.96*2),(a.shape[0],))
var_b_inf_mcmc = std_B_inf_mcmc
var_b_10_vae = np.reshape(std_b_10_vae*(1.96*2),(a.shape[0],))
var_b_10_avb = np.reshape(std_b_10_avb*(1.96*2),(a.shape[0],))
var_b_10_mcmc = std_B_10_mcmc
var_b_5_vae = np.reshape(std_b_5_vae*(1.96*2),(a.shape[0],))
var_b_5_avb = np.reshape(std_b_5_avb*(1.96*2),(a.shape[0],))
var_b_5_mcmc = std_B_5_mcmc
var_b_2_vae = np.reshape(std_b_2_vae*(1.96*2),(a.shape[0],))
var_b_2_avb = np.reshape(std_b_2_avb*(1.96*2),(a.shape[0],))
var_b_2_mcmc = std_B_2_mcmc

data_vae = [var_b_inf_vae.tolist(), var_b_10_vae.tolist(), var_b_5_vae.tolist(), var_b_2_vae.tolist()]
data_avb = [var_b_inf_avb.tolist(), var_b_10_avb.tolist(), var_b_5_avb.tolist(), var_b_2_avb.tolist()]
data_mcmc = [var_b_inf_mcmc.tolist(), var_b_10_mcmc.tolist(), var_b_5_mcmc.tolist(), var_b_2_mcmc.tolist()]
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
plt.ylim(0, 5)
plt.title('95% CI of Amplitude B, Bi-exponential Model')
plt.ylabel('95% CI')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('var_B_biexp.png')



######################## Compare Noise 
########### 1) Mean value
diff_e_inf_vae = np.reshape(mean_e_inf_vae.reshape((k,))-0,(e_std.shape[0],))
diff_e_inf_avb = np.reshape(mean_e_inf_avb.reshape((k,))-0,(e_std.shape[0],))
diff_e_inf_mcmc = np.reshape(mean_e_inf_mcmc.reshape((k,))-0,(e_std.shape[0],))
diff_e_10_vae = np.reshape(mean_e_10_vae.reshape((k,))-0.9/10,(e_std.shape[0],))
diff_e_10_avb = np.reshape(mean_e_10_avb.reshape((k,))-0.9/10,(e_std.shape[0],))
diff_e_10_mcmc = np.reshape(mean_e_10_mcmc.reshape((k,))-0.9/10,(e_std.shape[0],))
diff_e_5_vae = np.reshape(mean_e_5_vae.reshape((k,))-0.9/5,(e_std.shape[0],))
diff_e_5_avb = np.reshape(mean_e_5_avb.reshape((k,))-0.9/5,(e_std.shape[0],))
diff_e_5_mcmc = np.reshape(mean_e_5_mcmc.reshape((k,))-0.9/5,(e_std.shape[0],))
diff_e_2_vae = np.reshape(mean_e_2_vae.reshape((k,))-0.9/2.5,(e_std.shape[0],))
diff_e_2_avb = np.reshape(mean_e_2_avb.reshape((k,))-0.9/2.5,(e_std.shape[0],))
diff_e_2_mcmc = np.reshape(mean_e_2_mcmc.reshape((k,))-0.9/2.5,(e_std.shape[0],))

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
plt.ylim(-0.4, 0.4)
plt.title('Error in Noise Parameter, Bi-exponential Model')
plt.ylabel('Error')
plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('mean_e_biexp.png')

