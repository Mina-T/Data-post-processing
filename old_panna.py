import numpy as np
import pandas as pd
import os
import pickle


def F_MAE(_file):
    df = pd.read_csv(_file, delim_whitespace=True)
    ex = abs(df['fx_ref'] - df['fx_nn'])
    ey = abs(df['fy_ref'] - df['fy_nn'])
    ez = abs(df['fz_ref'] - df['fz_nn'])
    FMAE = sum(ex+ ey+ez)/len(df)
    return FMAE 


def force_file(name):
    '''
    Filters force files
    '''
    if name.endswith('_forces.dat'):
        return True
    else:
        return False


def validation_F_MAE(path):
    '''
    validate F over many epochs
    '''
    os.chdir(path)
    all_files = os.listdir()
    validation_f_MAE = []
    force_files = list(filter(force_file, all_files))
    x_F = sorted([int(file[6:9].replace('_', '').replace('s', '')) for file in force_files])
    files = sorted([s for s in force_files], key = lambda x: int(x[6:9].replace('_', '').replace('s', '')))
    for f in files:
        df = pd.read_csv(path + '/' + f, delim_whitespace = True)
        FMAE = F_MAE(f)
        validation_f_MAE.append(FMAE)
    
    return x_F, validation_f_MAE

def energy_file(name):
    '''
    Filters Energy files
    '''
    if name.endswith('0.dat'):
        return True
    else:
        return False

def validation_MAE(path):
    '''
    validate E over many epochs
    '''
    os.chdir(path)
    all_files = os.listdir()
    validation_MAE = []
    energy_files = list(filter(energy_file, all_files))
    x_E = sorted([int(file[6:9].replace('_', '').replace('s', '')) for file in energy_files])
    E_files = sorted([s for s in energy_files], key = lambda x: int(x[6:9].replace('_', '').replace('s', '')))
    for f in E_files:
        df = pd.read_csv(path + '/' + f, delim_whitespace = True)
        diff = abs(df['e_nn']-df['e_ref'])/ df['n_atoms']
        avg = sum(diff)/len(df)
        validation_MAE.append(avg)
    return x_E, validation_MAE
