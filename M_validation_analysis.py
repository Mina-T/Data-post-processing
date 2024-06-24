import pandas as pd
import os
import glob
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
from M_dataframe import read_metrics, read_dat


class check_validation:
    def __init__(self, df, freq_cut=0, err_cut=0):
        self.df = df
        self.freq_cut = freq_cut
        self.err_cut = err_cut

    def config_freq(self):
        configs = {}
        for idx, row in self.df.iterrows():
            Id = row['#filename'].split('-')[-1]
            if Id in configs.keys():
                configs[Id] = configs[Id] + 1
            else:
                configs[Id]= 1
        return configs

    def cut_freq(self):
        conf = self.config_freq()
        for k in conf.keys():
            if conf[k] == self.freq_cut:
                print(f'frequency of {k} is more than {self.freq_cut}')
    def F_MAE(self):
        Ex = abs(self.df['fx_nn'] - self.df['fx_ref'])
        Ey = abs(self.df['fy_nn'] - self.df['fy_ref'])
        Ez = abs(self.df['fz_nn'] - self.df['fz_ref'])
        Err_tot = sum(Ex + Ey + Ez)
        print(f'MAE of force on each atom is {(Err_tot / (len(self.df)*3))}')
        return Err_tot / (len(self.df)*3)
        
    def F_RMSE(self):
        sum_comp = sum((self.df['fx_nn'] - self.df['fx_ref'])**2 + (self.df['fy_nn'] - self.df['fy_ref'])**2  + (self.df['fz_nn'] - self.df['fz_ref'])**2 )
        avg = (sum_comp / (3 * len(self.df)))
        rmse =  (avg)**0.5
        print(rmse)
        return rmse
     

    def E_MAE(self, EwT=0.02):
        ids_e = {}
        EFwT = 0
        for idx, row in self.df.iterrows():
            Id = row['#filename']
            ids_e[Id] = abs(row['e_ref']-row['e_nn'])
            if abs(row['e_ref']-row['e_nn']) < EwT:
                EFwT+=1
        atoms = sum(self.df['n_atoms'])
        print(f'There are {len(ids_e.values())} structures and MAE/atm is {sum(ids_e.values())/atoms} eV')
        print(f'Average MAE of each structure is {sum(ids_e.values())/len(ids_e.values())}')
        print(f'{(EFwT/len(ids_e.keys()))*100} % within E threshold.' )
        return sum(ids_e.values())/len(ids_e.values())

    def check_hierarchy(self):
        '''
        Returns the number of systems for which the model has + OR - error values
        '''
        ids = {}
        for idx, row in self.df.iterrows():
            Id = row['#filename'].split('_')[-1]
            if Id not in ids.keys():
                ids[Id] = {}
                ids[Id]['e_nn'] = []
                ids[Id]['e_ref'] = []
            ids[Id]['e_nn'].append(row['e_nn'])
            ids[Id]['e_ref'].append(row['e_ref'])
        hierarchy_dict = {}
        c = 0
        h_id = []
        n = 0
        tot = 0
        for k in ids.keys():
            if len(ids[k]['e_nn']) > 1:
                tot += 1
                if k not in hierarchy_dict.keys():
                    hierarchy_dict[k] = []
                e_nn = ids[k]['e_nn']
                e_ref = ids[k]['e_ref']
                new_e_nn = [[idx, item ] for idx, item in enumerate(e_nn)]
                sorted_e_nn = sorted(new_e_nn, key = lambda x : x[1])
                new_e_ref = [[idx, item ] for idx, item in enumerate(e_ref)]
                sorted_e_ref = sorted(new_e_ref, key = lambda x : x[1])
                idx_nn = [i[0] for i in sorted_e_nn]
                idx_ref = [i[0] for i in sorted_e_ref]
                min_nn = min([i[1] for i in sorted_e_nn])
                min_ref = min([i[1] for i in sorted_e_ref])
                delta_ref = [r - min_ref for r in [i[1] for i in sorted_e_ref]]
                delta_nn = [n - min_nn for n in [i[1] for i in sorted_e_nn]]
                err = [abs(r - n) for r, n in zip(delta_ref, delta_nn)]
                if idx_nn == idx_ref: # and all(ele < 0.5 for ele in err)
                    h_id.append(k)
                    c += 1
                else:
                    n += 1
        print(f'out of {tot} systems with more than one configuration, hierarchy is conserved in {c} of them, NOT in {n} systems')
        return c

def E_MAE(df):
    mae = abs((df['e_ref']/df['n_atoms']) - (df['e_nn']/df['n_atoms']))
    mae = sum(mae)/len(mae)
    return mae

def F_MAE(df):
        Ex = abs(df['fx_nn'] - df['fx_ref'])
        Ey = abs(df['fy_nn'] - df['fy_ref'])
        Ez = abs(df['fz_nn'] - df['fz_ref'])
        Err_tot = sum(Ex + Ey + Ez)
        mae = Err_tot /(len(df)*3)
        return mae


def RMSE_E(df):
    diff = (df['e_ref'] - df['e_nn'])**2
    avg = sum(diff)/(len(diff)*sum(df['n_atoms']))
    rmse = avg**0.5
    print('RMSE is ',rmse, ' per atom')
    return rmse


           
class Read_validation:
    def __init__(self, _path, dirs, sub_dirs):
        self._path = _path
        self.dirs = dirs
        self.sub_dirs = sub_dirs





class Read_domains_validation:
    def __init__(self, _path, dirs, sub_dirs, epoch):
        self._path = _path
        self.dirs = dirs
        self.sub_dirs = sub_dirs
        self.epoch = epoch

    def get_error(self):
        model_validation = {_dir:{'E_MAE':[], 'F_MAE':[]} for _dir in self.dirs}
        for _dir in self.dirs:
            for s_dir in self.sub_dirs:
                try:
                    print(_dir, s_dir, flush = True)
                    p = os.path.join(self._path, _dir, s_dir, 'val/')
                    os.chdir(p)
                    energy_file = glob.glob(f"epoch_{self.epoch}_step_{self.epoch*1000}.dat")
                    energy_file = read_dat(p, energy_file[0])
                    e_MAE = E_MAE(energy_file)
                    model_validation[_dir]['E_MAE'].append(e_MAE)
                    force_file = glob.glob(f"epoch_{self.epoch}_step_{self.epoch*1000}_forces.dat")
                    force_file = read_dat(p, force_file[0])
                    f_MAE = F_MAE(force_file)
                    model_validation[_dir]['F_MAE'].append(f_MAE)
                    print(f'MAE_E: {e_MAE}, MAE_F: {f_MAE}', flush = True)
                except:
                    print('file not found', flush = True)
                    continue
            if model_validation[_dir]['E_MAE']:
                model_validation[_dir]['E_MAE'] = sum(model_validation[_dir]['E_MAE'])/len(model_validation[_dir]['E_MAE'])
            if model_validation[_dir]['F_MAE']:
                model_validation[_dir]['F_MAE'] = sum(model_validation[_dir]['F_MAE'])/len(model_validation[_dir]['F_MAE'])
        print(model_validation, flush = True)
        return model_validation

