from random import randint  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from M_validation_analysis import *
from M_dataframe import read_metrics, read_dat


def plot_df(df, axs=None ,label=None, color=None, factor = 1, validation = True, train = False):
    try:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            try:
                axs[0].plot(df['#epoch'] * factor, df['VMAE/at'], color= color, alpha = 0.8 , ms = 6.0, label = label)
                axs[1].plot(df['#epoch'] * factor, df['VMAEF'], color= color, alpha = 0.8, ms = 6.0) #+  ' V'
            except:
                axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.8 , ms = 6.0, label = label)
                axs[1].plot(df['#epoch'] * factor, df['Val_MAEF'], color= color, alpha = 0.8, ms = 6.0)
        if train == True:
                axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08)
                axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.5, ms = 0.08)
    except:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.8 , ms = 6.0, label = label)
            axs[1].plot(df['#epoch'] * factor, df['Val_MAEF'], color= color, alpha = 0.8, ms = 6.0) #+  ' V'
        if train == True:
            axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08)
            axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.5, ms = 0.08)


def plot_E_train(df, axs=None ,label=None, color=None, factor = 1, validation = True, train = False):
    try:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            try:
                axs[0].plot(df['#epoch'] * factor, df['VMAE/at'], color= color, alpha = 0.8 , ms = 6.0,  label = label)
            except:
                axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.8 , ms = 6.0,  label = label)
        if train == True:
                axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08,  label = label)
    except:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.8 , ms = 6.0,  label = label)
        if train == True:
            axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.5, ms = 0.08,  label = label)


#def plot_replica_noise(all_dirs, labels, axs = None, validation =True,  train = False):
 #   for noise, model in zip(all_dirs, labels):
 #       color = ('#%06X' % randint(0, 0xFFFFFF))
 #       if train:
 #           axs[0].plot([noise[epoch][model]['MAE/at'] for epoch in noise.keys()], color = color, label = label+' T', alpha = 0.5)
 #           axs[1].plot([noise[epoch][model]['MAEF'] for epoch in noise.keys()], color = color, label = label+' T', alpha = 0.5)
 #       if validation:
 #           axs[0].plot([noise[epoch][model]['VMAE/at'] for epoch in noise.keys()], color = color, label = label)
 #           axs[1].plot([noise[epoch][model]['VMAEF'] for epoch in noise.keys()], color = color, label = label)

def plot_validation(path, _file, label):
    f = read_dat(path, _file)
    plt.plot(f['e_ref']/f['n_atoms'], f['e_nn']/f['n_atoms'], '.r', label = label)
    plt.xlabel('E ref', size = 16)
    plt.ylabel('E nn', size = 16)
    plt.legend()
    plt.show()


def plot_setup(fig, axs, log = True, lim = True, x0lim = 10,  xlim = 2500):
    axs[0].legend(loc = 'upper right', prop={'size': 12})
    axs[0].set_ylabel('MAE')
    fig.supxlabel('log Epochs(1000 steps each)')
    if lim:
        axs[0].set_xlim(x0lim,xlim)
        axs[0].set_ylim(0.01, 0.06)
        axs[1].set_xlim(x0lim,xlim)
        axs[1].set_ylim(0.08, 0.16)

    minor_ticks = np.arange(0, 0.1, 0.005)
    axs[0].set_yticks(minor_ticks, minor=True)
    axs[0].grid(which='minor', axis = 'y', alpha=0.2)
    axs[0].grid(which='major', axis = 'y', alpha=0.7)

    minor_ticks = np.arange(0.05, 0.2, 0.005)
    axs[1].set_yticks(minor_ticks, minor=True)
    axs[1].grid(which='minor', axis = 'y', alpha=0.2)
    axs[1].grid(which='major', axis = 'y', alpha=0.7)
    if log:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
    plt.show()


def plot_learning_curve(labels:list, h_t, h_in, h_out, emb_t, emb_in, emb_out):
    x = [i for i in range(1, len(labels)+1)]
    plt.plot(x, emb_t, 'r', label ='Training')
    plt.plot(x, emb_in, 'c', label ='In domain')
    plt.plot(x, emb_out, 'g', label = 'Out of domain')

    plt.plot(x, h_t, '-r',linestyle='dashed')
    plt.plot(x, h_in, '-c', linestyle='dashed')
    plt.plot(x, h_out, '-g', linestyle='dashed')

    plt.ylim(-0.005, 0.13)
    plt.xticks(ticks = x, labels = labels)
    plt.xlabel('Training set size', fontsize = 12)
    plt.ylabel('Error (meV/atm)', fontsize = 12)
    plt.legend()
    plt.show()

