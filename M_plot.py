from random import randint  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from M_validation_analysis import check_validation, replicas_error
from M_dataframe import read_metrics, read_dat


def plot_df(df, axs=None ,label=None, color=None, factor = 1, validation = True, train = False):
    if not color:
        color = ('#%06X' % randint(0, 0xFFFFFF))
    if validation == True:
        axs[0].plot(df['#epoch'] * factor, df['VMAE/at'], color= color, alpha = 0.8 , ms = 6.0)
        axs[1].plot(df['#epoch'] * factor, df['VMAEF'], color= color, alpha = 0.8, ms = 6.0, label = label) #+  ' V'
    if train == True:
        axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08,  label = label)
        axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.5, ms = 0.08)


def plot_replica_noise(all_dirs, labels, axs = None, validation =True,  train = False):
    for noise, model in zip(all_dirs, labels):
        color = ('#%06X' % randint(0, 0xFFFFFF))
        if train:
            axs[0].plot([noise[epoch][model]['MAE/at'] for epoch in noise.keys()], color = color, label = label+' T', alpha = 0.5)
            axs[1].plot([noise[epoch][model]['MAEF'] for epoch in noise.keys()], color = color, label = label+' T', alpha = 0.5)
        if validation:
            axs[0].plot([noise[epoch][model]['VMAE/at'] for epoch in noise.keys()], color = color, label = label)
            axs[1].plot([noise[epoch][model]['VMAEF'] for epoch in noise.keys()], color = color, label = label)

def plot_validation(path, _file, label):
    f = read_dat(path, _file)
    plt.plot(f['e_ref']/f['n_atoms'], f['e_nn']/f['n_atoms'], '.r', label = label)
    plt.xlabel('E ref', size = 16)
    plt.ylabel('E nn', size = 16)
    plt.legend()
    plt.show()
