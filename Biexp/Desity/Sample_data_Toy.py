import os.path
import subprocess
from termios import B150
import numpy as np
import sklearn.metrics
import pandas as pd
import nibabel as nib
from svb import DataModel
from svb_models_asl import AslRestModel, AslNNModel
import logging
logging.getLogger().setLevel(logging.INFO)
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from svb.main import run
from sklearn.metrics import r2_score
import sklearn



# Infer a and B, which are rate1 and amp2
# a = rate1
# b = rate2
# A = amp1
# B = amp2



n = 121
rd_state = np.random.RandomState(25)
#tpts = rd_state.uniform(1.0, 5.0, size=(n,))        
A = np.repeat(1,n)
B = np.repeat(np.linspace(0.5,1.5,11),11)
a = np.tile(np.linspace(0.5,1.5,11),(1,11))
b = np.repeat(1,n)
e_std = np.zeros((n,1),np.float32)
e_log = np.log(e_std)



T = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
TPS = np.tile(T,(n,1))
A1 = np.tile(A.reshape(n,1),(1,6))
a1 = np.tile(a.reshape(n,1),(1,6))
B1 = np.tile(B.reshape(n,1),(1,6))
b1 = np.tile(b.reshape(n,1),(1,6))


def signal(t, A, B, a, b):
    signal = A*np.exp(-a*t) + B*np.exp(-b*t)
    return signal


signal = signal(TPS, A1, B1, a1, b1)

df_a = pd.DataFrame(a)
df_b = pd.DataFrame(b)
df_A = pd.DataFrame(A)
df_B = pd.DataFrame(B)
df_signal = pd.DataFrame(signal)
df_a.to_csv('little_a.csv')
df_b.to_csv('little_b.csv')
df_A.to_csv('A.csv')
df_B.to_csv('B.csv')
df_signal.to_csv('signal.csv')


for snr in (1000000, 10, 5, 2.5):
    e_std = np.repeat(0.9/snr,n)
    e_log = np.log(e_std)
    print("Noise STD: ", e_std)
    signal_noise = np.random.normal(signal, np.tile(np.reshape(e_std,(n,1)),(1,len(T))))
    df_e_log = pd.DataFrame(e_log)
    df_e_std = pd.DataFrame(e_std)
    df_signal_noise = pd.DataFrame(signal_noise)
    df_e_log.to_csv('e_log.csv')
    df_e_std.to_csv('e_std.csv')
    df_signal_noise.to_csv('signal_noise_snr_%i.csv' % (snr*10))


