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

outdir_0 = "out_0"
outdir_1 = "out_1"
outdir_2 = "out_2"
outdir_3 = "out_3"
outdir_4 = "out_4"
outdir_5 = "out_5"
outdir_6 = "out_6"
outdir_7 = "out_7"

j = 10

scaling = 200/(1090*0.85)
mask = nib.load('mask.nii.gz')
mask = np.array(mask.dataobj)[:,:,j:j+1]

mean_f_avb_0 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_0).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_0 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_e_avb_0 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_0).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_0[np.where(mask == 0)] = 0
std_f_avb_0 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_0).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_0 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_0 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_0).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_0[np.where(mask == 0)] = 0
mean_s_avb_0 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_sp_avb_0 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_0).get_data())[:,:,j:j+1]
mean_p_avb_0 = mean_sp_avb_0/(mean_s_avb_0+10**-6)
std_s_avb_0 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_0 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_0).get_data())*(1.96*2))[:,:,j:j+1]



mean_f_avb_1 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_1).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_1 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_1).get_data())[:,:,j:j+1]
mean_e_avb_1 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_1).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_1[np.where(mask == 0)] = 0
std_f_avb_1 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_1).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_1 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_1).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_1 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_1).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_1[np.where(mask == 0)] = 0
mean_s_avb_1 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_1).get_data())[:,:,j:j+1]
mean_sp_avb_1 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_1).get_data())[:,:,j:j+1]
mean_p_avb_1 = mean_sp_avb_1/(mean_s_avb_1+10**-6)
std_s_avb_1 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_1).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_1 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_1).get_data())*(1.96*2))[:,:,j:j+1]


mean_f_avb_2 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_2).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_2 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_2).get_data())[:,:,j:j+1]
mean_e_avb_2 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_2).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_2[np.where(mask == 0)] = 0
std_f_avb_2 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_2).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_2 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_2).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_2 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_2).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_2[np.where(mask == 0)] = 0
mean_s_avb_2 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_2).get_data())[:,:,j:j+1]
mean_sp_avb_2 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_2).get_data())[:,:,j:j+1]
mean_p_avb_2 = mean_sp_avb_2/(mean_s_avb_2+10**-6)
std_s_avb_2 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_2).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_2 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_2).get_data())*(1.96*2))[:,:,j:j+1]


mean_f_avb_3 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_3).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_3 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_3).get_data())[:,:,j:j+1]
mean_e_avb_3 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_3).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_3[np.where(mask == 0)] = 0
std_f_avb_3 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_3).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_3 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_3).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_3 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_3).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_3[np.where(mask == 0)] = 0
mean_s_avb_3 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_3).get_data())[:,:,j:j+1]
mean_sp_avb_3 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_3).get_data())[:,:,j:j+1]
mean_p_avb_3 = mean_sp_avb_3/(mean_s_avb_3+10**-6)
std_s_avb_3 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_3).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_3 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_3).get_data())*(1.96*2))[:,:,j:j+1]


mean_f_avb_4 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_4).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_4 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_4).get_data())[:,:,j:j+1]
mean_e_avb_4 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_4).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_4[np.where(mask == 0)] = 0
std_f_avb_4 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_4).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_4 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_4).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_4 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_4).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_4[np.where(mask == 0)] = 0
mean_s_avb_4 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_4).get_data())[:,:,j:j+1]
mean_sp_avb_4 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_4).get_data())[:,:,j:j+1]
mean_p_avb_4 = mean_sp_avb_4/(mean_s_avb_4+10**-6)
std_s_avb_4 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_4).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_4 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_4).get_data())*(1.96*2))[:,:,j:j+1]


mean_f_avb_5 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_5).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_5 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_5).get_data())[:,:,j:j+1]
mean_e_avb_5 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_5).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_5[np.where(mask == 0)] = 0
std_f_avb_5 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_5).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_5 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_5).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_5 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_5).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_5[np.where(mask == 0)] = 0
mean_s_avb_5 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_5).get_data())[:,:,j:j+1]
mean_sp_avb_5 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_5).get_data())[:,:,j:j+1]
mean_p_avb_5 = mean_sp_avb_5/(mean_s_avb_5+10**-6)
std_s_avb_5 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_5).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_5 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_5).get_data())*(1.96*2))[:,:,j:j+1]


mean_f_avb_6 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_6).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_6 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_6).get_data())[:,:,j:j+1]
mean_e_avb_6 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_6).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_6[np.where(mask == 0)] = 0
std_f_avb_6 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_6).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_6 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_6).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_6 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_6).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_6[np.where(mask == 0)] = 0
mean_s_avb_6 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_6).get_data())[:,:,j:j+1]
mean_sp_avb_6 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_6).get_data())[:,:,j:j+1]
mean_p_avb_6 = mean_sp_avb_6/(mean_s_avb_6+10**-6)
std_s_avb_6 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_6).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_6 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_6).get_data())*(1.96*2))[:,:,j:j+1]



mean_f_avb_7 = (nib.load("%s/step2/mean_ftiss.nii.gz" % outdir_7).get_data()*scaling)[:,:,j:j+1]
mean_att_avb_7 = (nib.load("%s/step2/mean_delttiss.nii.gz" % outdir_7).get_data())[:,:,j:j+1]
mean_e_avb_7 = (1/(nib.load("%s/step2/noise_means.nii.gz" % outdir_7).get_data()+10**-6)*scaling)[:,:,j:j+1]
mean_e_avb_7[np.where(mask == 0)] = 0
std_f_avb_7 = ((nib.load("%s/step2/std_ftiss.nii.gz" % outdir_7).get_data())*(1.96*2)*scaling)[:,:,j:j+1]
std_att_avb_7 = ((nib.load("%s/step2/std_delttiss.nii.gz" % outdir_7).get_data())*(1.96*2))[:,:,j:j+1]
std_e_avb_7 = (1/(nib.load("%s/step2/noise_stdevs.nii.gz" % outdir_7).get_data())+10**-6)[:,:,j:j+1]
std_e_avb_7[np.where(mask == 0)] = 0
mean_s_avb_7 = (nib.load("%s/step2/mean_disp1.nii.gz" % outdir_7).get_data())[:,:,j:j+1]
mean_sp_avb_7 = (nib.load("%s/step2/mean_disp2.nii.gz" % outdir_7).get_data())[:,:,j:j+1]
mean_p_avb_7 = mean_sp_avb_7/(mean_s_avb_7+10**-6)
std_s_avb_7 = ((nib.load("%s/step2/std_disp1.nii.gz" % outdir_7).get_data())*(1.96*2))[:,:,j:j+1]
std_sp_avb_7 = ((nib.load("%s/step2/std_disp2.nii.gz" % outdir_7).get_data())*(1.96*2))[:,:,j:j+1]












######################## VAE

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(e) Estimated Perfusion by aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90(mean_f_avb_0*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90(mean_f_avb_1*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90(mean_f_avb_2*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90(mean_f_avb_3*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(mean_f_avb_4*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(mean_f_avb_5*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(mean_f_avb_6*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(mean_f_avb_7*100,k=1,axes=(0,1)), cmap = "hot",vmax=175,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_perfusion_single_repeat.png')
#plt.show()


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(g) Estimated ATT by aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90(mean_att_avb_0,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90(mean_att_avb_1,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90(mean_att_avb_2,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90(mean_att_avb_3,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(mean_att_avb_4,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(mean_att_avb_5,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(mean_att_avb_6,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(mean_att_avb_7,k=1,axes=(0,1)), cmap = "hot",vmax=3,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_att_single_repeat.png')
#plt.show()



fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(e) 95% CI of Perfusion, aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90(std_f_avb_0*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90(std_f_avb_1*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90(std_f_avb_2*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90(std_f_avb_3*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(std_f_avb_4*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(std_f_avb_5*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(std_f_avb_6*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(std_f_avb_7*100,k=1,axes=(0,1)), cmap = "Purples",vmax=200,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_perfusion_std_single_repeat.png')
#plt.show()





fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('(g) 95% CI of ATT, aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90(std_att_avb_0,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90(std_att_avb_1,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90(std_att_avb_2,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90(std_att_avb_3,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90(std_att_avb_4,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90(std_att_avb_5,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90(std_att_avb_6,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90(std_att_avb_7,k=1,axes=(0,1)), cmap = "Purples",vmax=2,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_att_std_single_repeat.png')
#plt.show()




fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('Estimated Sharpness by aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90((mean_s_avb_0),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90((mean_s_avb_1),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90((mean_s_avb_2),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90((mean_s_avb_3),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90((mean_s_avb_4),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90((mean_s_avb_5),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90((mean_s_avb_6),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90((mean_s_avb_7),k=1,axes=(0,1)), cmap = "hot",vmax=10,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_Sharpness_single_repeat.png')
#plt.show()


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('Estimated Time-to-peak by aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90((mean_p_avb_0),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90((mean_p_avb_1),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90((mean_p_avb_2),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90((mean_p_avb_3),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90((mean_p_avb_4),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90((mean_p_avb_5),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90((mean_p_avb_6),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90((mean_p_avb_7),k=1,axes=(0,1)), cmap = "hot",vmax=0.25,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_Time-to-peak_single_repeat.png')
#plt.show()




fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(13, 6))
#fig.set_size_inches(14, 6)
fig.suptitle('Estimated Noise Parameter by aVB, Dispersion KM, Reduced Data Sets', fontsize=16, y = 0.96)
#haha.set_position([.5, 1.05])
im0_avb = axes[0,0].imshow(np.rot90((mean_e_avb_0*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im1_avb = axes[0,1].imshow(np.rot90((mean_e_avb_1*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im2_avb = axes[0,2].imshow(np.rot90((mean_e_avb_2*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im3_avb = axes[0,3].imshow(np.rot90((mean_e_avb_3*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
axes[0,0].set_title('Repeat 1',fontsize=10)
axes[0,1].set_title('Repeat 2',fontsize=10)
axes[0,2].set_title('Repeat 3',fontsize=10)
axes[0,3].set_title('Repeat 4',fontsize=10)
im0_avb = axes[1,0].imshow(np.rot90((mean_e_avb_4*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im1_avb = axes[1,1].imshow(np.rot90((mean_e_avb_5*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im2_avb = axes[1,2].imshow(np.rot90((mean_e_avb_6*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
im3_avb = axes[1,3].imshow(np.rot90((mean_e_avb_7*100),k=1,axes=(0,1)), cmap = "hot",vmax=150,vmin=0)
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
cbar = fig.colorbar(im3_avb, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
fig.tight_layout(pad=1.5)
plt.savefig('real_data_Std of Error Term_single_repeat.png')
#plt.show()




