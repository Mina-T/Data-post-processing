import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from M_dataframe import read_metrics


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

class replicas_error:
    def __init__(self, dirs, label):
        self.dirs = dirs
        self.label = label

    def get_replicas_TV_err(self):
        '''
        returns [{epoch: {MAE/at:err}} * replicas]
        '''
        TV_err = []
        c = 0
        for replica in self.dirs:
            errs = {}
            df = read_metrics(replica)
            for idx, epoch  in df.iterrows():
                if idx not in errs.keys():
                   errs[idx] = {}
                errs[idx]['MAE/at'] = epoch['MAE/at']
                errs[idx]['VMAE/at'] = epoch['VMAE/at'] 
                errs[idx]['MAEF'] = epoch['MAEF']
                errs[idx]['VMAEF'] = epoch['VMAEF']
            TV_err.append(errs)
        return TV_err

    def avg_model(self):
        '''
        returns avg = {epoch: {MAE/at:avg_err}}
        '''
        TV_errors = self.get_replicas_TV_err()  # [{epoch: {MAE/at:err}} * replicas]
        avg = {}
        n_replicas = len(TV_errors)
        epochs = min([replica.keys() for replica in TV_errors])
        print('min epochs: ', epochs)
        for epoch in range(epochs):
            avg[epoch] = {}
            MAE_at, MAEF, VMAE_at, VMAEF = 0,0,0,0
            for replica in TV_errors:
                MAE_at += replica[epoch]['MAE/at']
                MAEF += replica[epoch]['MAEF']
                VMAE_at += replica[epoch]['VMAE/at']
                VMAEF += replica[epoch]['VMAEF']
            avg[epoch]['MAE/at'], avg[epoch]['MAEF'], avg[epoch]['VMAE/at'],  avg[epoch]['VMAEF'] = MAE_at/n_replicas, MAEF/n_replicas , VMAE_at/n_replicas, VMAEF/n_replicas
            
        return avg

    def get_noise(self):
        '''
        returns noise = {replica :{epoch :{'MAE/at' : noise}}}
        '''
        noise = {}
        avg = self.avg_model() # {epoch: {MAE/at:avg_err}}
        TV_errors = self.get_replicas_TV_err()# [{epoch: {MAE/at:err}} * replicas]
        replica_c = 0
        for replica in TV_errors:
            noise[replica_c] = {}
            for epoch in avg.keys():
                noise[replica_c][epoch] = {}
                noise[replica_c][epoch]['MAE/at'] = replica[epoch]['MAE/at'] - avg[epoch]['MAE/at']
                noise[replica_c][epoch]['VMAE/at'] = replica[epoch]['VMAE/at'] - avg[epoch]['VMAE/at']
                noise[replica_c][epoch]['MAEF'] = replica[epoch]['MAEF'] - avg[epoch]['MAEF']
                noise[replica_c][epoch]['VMAEF'] = replica[epoch]['VMAEF'] - avg[epoch]['VMAEF']
            replica_c+=1
        return noise

    def config_bias(self, epochs:list, path:str):
        '''
        Returns {row_Id:[b1,..,bn]}
        '''
        biases = {}
        for epoch in epochs:
            biases[epoch] = {}
            for model in self.dirs:
                os.chdir(path + str(model))
                print('checking ', os.getcwd())
                df = pd.read_csv(f'epoch_{epoch}_step_{1000*epoch}.dat', delim_whitespace = True)
                for idx, row in df.iterrows():
                    if row['#filename'] not in biases[epoch].keys():
                            biases[epoch][row['#filename']] = []
                    biases[epoch][row['#filename']].append(row['e_nn']- row['e_ref']) 
        avg_bias = {}
        for Id in biases.keys():
                avg_bias[Id] = {}
                for epoch in biases[Id].keys():
                    avg_bias[Id][epoch] = sum(biases[Id][epoch])/len(biases[Id][epoch])
        return biases, avg_bias


    def config_variance(self, biases, avg_bias):
        '''
        Returns a dict of systems id and their variance
        '''
        V_dict = {}
        for epoch in biases.keys(): 
            V_dict[epoch] = {}
            for Id in biases[epoch].keys():
                avg = avg_bias[Id][epoch]
                diffs = []
                for err in biases[epoch][Id]:
                    diffs.append((err - avg)**2)
                var = sum(diffs)/ len(diffs)   
                V_dict[epoch][Id] = var
        return V_dict


def bias(df):
    #df = pd.read_csv(_file, delim_whitespace = True)
    err = (df['e_ref']/df['n_atoms']) - (df['e_nn']/df['n_atoms'])
    bias = sum(err)/len(err)
    print(f'bias is {bias} per atom')
    return bias

def variance(df):
    #df = pd.read_csv(_file, delim_whitespace = True)
    err = df['e_ref'] - df['e_nn']
    avg = sum(err)/len(err)
    diff = (err - avg)**2
    var = sum(diff)/len(diff)
    return var

def MAE(df):
    mae = abs((df['e_ref']/df['n_atoms']) - (df['e_nn']/df['n_atoms']))
    mae = sum(mae)/len(mae)
    print(f'MAE is {mae} per atom')
    return mae

# def read_mace_validation():
# mace_energies = []
# p = '/leonardo_scratch/large/userexternal/mtaleblo/MACE/validation/3BPA/'
# f = open(p+'output.xyz', 'r')
# idx = 0
# while True:
#     try:
#         structure = list(extxyz.read_xyz(f, index = idx, properties_parser = extxyz.key_val_str_to_dict))
#         mace_energies.append([idx, len(structure[0].get_atomic_numbers()),structure[0].info['energy'] , structure[0].info['MACE_energy']])
#         idx += 1
#         if idx %1000 == 0 :
#             print(idx)
#     except:
#         print('Done')
#         break
# mace_energies = np.array(mace_energies)
# np.save(open(p+'mace_energies.npy', 'wb'), mace_energies)
