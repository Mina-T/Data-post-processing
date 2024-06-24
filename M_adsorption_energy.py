import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

class NN_AdsE:
    def __init__(self, slab_path, system_path, _file, adsorbate_energies_file, AdsE_ref_file):
        self.slab_path = slab_path
        self.system_path = system_path
        self.file = _file
        self.adsorbate_energies_file = adsorbate_energies_file
        self.AdsE_ref_file = AdsE_ref_file
        
    def get_adsorbate_energies(self): # return {id : adsorbateE}
        adsorbate_energies = pickle.load(open(self.adsorbate_energies_file, 'rb'))
        return adsorbate_energies

    def get_AdsE_ref(self):# return {id : Eads ref}
        AdsE_ref = pickle.load(open(self.AdsE_ref_file, 'rb'))
        return AdsE_ref
        
    def get_slab_energies(self): # return {id : slabE}
        os.chdir(self.slab_path)
        slabs = pd.read_csv(self.file, delim_whitespace = True)
        Slabs_energies = {}
        _id_slab = list(slabs.iloc[:,0])
        slab_energies = list(slabs.iloc[:,3])
        for _id, e in zip(_id_slab, slab_energies):
            Slabs_energies[_id] = e
        return Slabs_energies, _id_slab

    def get_system_energies(self): # return {id : nn_systemE} & {id : ref_systemE}
        os.chdir(self.system_path)
        systems = pd.read_csv(self.file, delim_whitespace = True)
        _sys_id = list(systems.iloc[:,0])
        Systems_energies = {}
        their_systems = {}
        systems_e = list(systems.iloc[:,3])
        ref_e = list(systems.iloc[:,2])
        for i, e, r in zip(_sys_id, systems_e, ref_e):
            Systems_energies[i] = e
            their_systems[i] = r
        return Systems_energies, their_systems
    
    def write_to_df(self): # returns {id: nn_AdsE}
        dict = {}
        adsorbate_energies = self.get_adsorbate_energies()
        AdsE_ref = self.get_AdsE_ref()
        Eslab_nn, slab_ids = self.get_slab_energies()
        Etot_nn , Etot_ref = self.get_system_energies()
        for i in slab_ids:
            dict[i] = {}
            Adsorbate_energy_i = adsorbate_energies[i]
            Etot_nn_i = Etot_nn[i]
            Eslab_nn_i = Eslab_nn[i]
            Etot_ref_i = Etot_ref[i]
            Adsorption_energy_i = Etot_nn_i - Eslab_nn_i - Adsorbate_energy_i
            dict[i]['E_adsorbate'] = Adsorbate_energy_i
            dict[i]['Etot_nn'] = Etot_nn_i
            dict[i]['Etot_ref'] = Etot_ref_i
            dict[i]['Eslab_nn'] = Eslab_nn_i
            dict[i]['Eslab_ref'] = Etot_ref_i - Adsorbate_energy_i
            dict[i]['AdsE_nn'] = Adsorption_energy_i
            dict[i]['AdsE_ref'] = AdsE_ref[i]

        df = pd.DataFrame.from_dict(dict, orient='index')
        return df      
