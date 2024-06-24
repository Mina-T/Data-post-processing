import os
import ase
from ase.io import extxyz
import numpy as np
import time
import pickle
import json
from ase.io import Trajectory as T
import numpy as np



ptable = ['null', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo']

#_map = pickle.load(open('/leonardo/home/userexternal/mtaleblo/s2ef/2M/oc20_data_mapping.pkl', 'rb'))

class Dataset:
    def __init__(self, _path, a, b):
        self.path = _path
        self.a = a
        self.b = b

    def extract_dataset(self):
        anomalies = 0
        for i in range(self.a, self.b):
            dataset = np.array([], dtype=object)
            dataset = np.hstack((dataset, np.array(['index = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))

            print(f'structure_{i}', flush = True)
            File = open(f'{i}.extxyz', 'r')
            txt_file = open(f'{i}.txt')
            read = [line for line in txt_file]
            Id_list = [line.split(',')[0].replace('random', '') for line in read]
            start = time.time()
            idx = 0
            while True:
                try:
                    structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
                    symbol = str(structure[0].symbols)
                    atoms = np.array(structure[0].get_chemical_symbols())
                    atoms = np.array( [ptable.index(j) for j in atoms])
                    natom = len(atoms)
                    cell = np.array(structure[0].get_cell())
                    pos = np.array(structure[0].get_positions())
                    energy = structure[0].get_total_energy()
                    force = np.array(structure[0].get_forces(apply_constraint=False))
        #            if _map[f'random{Id_list[idx]}']['anomaly'] == 0:
                    Id = str(i)+'_'+str(idx)+'_'+Id_list[idx]
                    lst = [idx, natom, atoms, cell, symbol, pos, energy, force, Id]
                    dataset = np.vstack((dataset, np.array(lst, dtype=object)))
                    idx+=1
                   # else:
                    #    anomalies +=1
                     #   idx +=1

                    if idx % 100 ==0 :
                           print(idx, flush = True)

                except:
                    write = open(f'val_{i}.npy', 'wb')
                    np.save(write, dataset)
                    break

        #print('anomalies: ', anomalies)
        end = time.time()
        duration = end - start
        print(duration, 'second')
        #write = open('moreV.npy', 'wb')
        #np.save(write, dataset)
        print('Done', flush = True)


def extract_dataset_mp(path, name):
    print(name , flush = True)
    os.chdir(path)
    files = os.listdir()
    files = [f for f in files if f.endswith('.extxyz')]
    counter = 0
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['mp_id= 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'task_id = 8'])))
 
    for i in files:
        name = i.replace('.extxyz', '')
        if counter % 1000 == 0:
            print(counter)
        counter +=1
        File = open(i, 'r')
        idx = 0
        while True:
                try:
                    structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
                    symbol = str(structure[0].symbols)
                    atoms = np.array(structure[0].get_chemical_symbols())
                    atoms = np.array( [ptable.index(j) for j in atoms])
                    natom = len(atoms)
                    cell = np.array(structure[0].get_cell())
                    pos = np.array(structure[0].get_positions())
                    energy = structure[0].info['corrected_total_energy']
                    force = np.array(structure[0].get_forces(apply_constraint=False))
                    Id = structure[0].info['task_id']
                    lst = [structure[0].info['mp_id'], natom, atoms, cell, symbol, pos, energy, force, Id]
                    dataset = np.vstack((dataset, np.array(lst, dtype=object)))
                    idx+=1
                except:
                    break

    write = open(f'{name}.npy', 'wb')
    np.save(write, dataset)
    print(i, ' Done', flush = True)


def extract_wbm_test_set(_file):
    f = json.load(open(_file, 'r'))
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['idx= 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
    
    material_ids = f['material_id'] # a dict
    material_formula = f['formula_from_cse']# a dict 
    systems = f['computed_structure_entry'] # a dict
    for _id in material_ids.keys():
        Id = _id
        structure = IStructure.from_dict(m['computed_structure_entry']['256962']['structure'])


def jmol_xyz_to_npy(path, name):
    '''
    converts jmol xyz files of QM9 dataset to npy files readable by LATTE
    '''
    atomic_ref = {1 : -0.500273,
                  6 : -37.846772,
                  7 : -54.583861,
                  8 :-75.064579,
                  9 : -99.718730}
    rm_list = np.load('/leonardo_scratch/large/userexternal/mtaleblo/Dataset/QM9/training_data/rm.npy', allow_pickle = True)
    print(rm_list[:3], flush = True)
    print(name, flush = True)
    os.chdir(path)
    _files = [f for f in os.listdir() if f.endswith('.xyz')]
    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['index = 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
    Hartree_to_eV = 27.2114
    for _file in _files:
        _id = _file.replace('dsgdb9nsd_', '').replace('.xyz', '')
        if _id in rm_list:
            print(_id, ' skipped', flush = True)
            continue
        else:
            f = open(_file, 'r')
            r = f.readlines()
            natoms = int(r[0].replace('\n', ''))
# axs[0].set_yscale('log')            properties = r[1].split('\t')
            properties.pop()
            properties[0:1] = properties[0].split(' ')
            Internal_energy_0K = float(properties[12])
            idx = properties[1]
            forces = np.zeros((natoms, 3))
            Id = properties[0]
            atomic_numbers = []
            atomization_energy = Internal_energy_0K - (sum([atomic_ref[n] for n in atomic_numbers]))
            symbol = ''
            atomic_positions = []
            for i in range(2, natoms+2):
                line = r[i].replace('\n', '').split('\t')
                atomic_numbers.append(ptable.index(line[0]))
                cell = np.array([[5.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, 5.0]])
                symbol += line[0]
                atomic_positions.append([float(n.replace(' ','').replace('*^','e')) for n in line[1:4]])
                lst = [idx, natoms, np.array(atomic_numbers), cell, symbol, np.array(atomic_positions), atomization_energy * Hartree_to_eV , forces, Id]
            dataset = np.vstack((dataset, np.array(lst, dtype=object)))
    write = open(f'{name}_AE.npy', 'wb')
    np.save(write, dataset)
    print('Done', flush = True)


def wbm_to_npy(path):
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.atoms import Atoms
    from pymatgen.core.structure import IStructure

    dataset = np.array([], dtype=object)
    dataset = np.hstack((dataset, np.array(['idx= 0', 'Number of atoms = 1', 'atomic numbers = 2','cell = 3', 'Chemical composition = 4', 'atomic positions = 5', 'Energy = 6', 'forces = 7', 'Id = 8'])))
    print('start', flush = True)
    for j in range(1,6):
        f = json.load(open(f'step_{j}.json', 'r'))
        f = f["entries"]
        for i in range(len(f)):
            material_id = 'wbm-' + str(j) + '-' + str(i)
            composition_dict = f[i]['composition'] # a dict
            system = f[i]['structure'] # a dict
            energy = f[i]['energy']
            natoms = int(sum(composition_dict.values()))
            atomic_numbers = []
            for k in composition_dict.keys():
                atomic_numbers.extend([ptable.index(k)]* int(composition_dict[k]))
            structure = AseAtomsAdaptor.get_atoms(IStructure.from_dict(system))
            pos = np.array(structure.get_positions())
            cell = np.array(structure.get_cell())
            forces = np.zeros((natoms, 3))
            symbol = str(structure.symbols)
            lst = [i, natoms, atomic_numbers, cell, symbol, pos,energy, forces, material_id]
            dataset = np.vstack((dataset, np.array(lst, dtype=object)))
            if i%5000==0:
                print(i, flush = True)

    write = open('wbm.npy', 'wb')
    np.save(write, dataset)
    print('Done', flush = True)


class Read_traj:

  def __init__(self, _file, step = -1):
    self.file = _file
    self.traj = T(_file)
    self.step = step

  def traj_length(self):
      return len(self.traj)

  def name(self):
    name =  self.file.replace('opt_', '').replace('.traj', '')
    print(name)
    return name

  def natoms(self):
    natoms = len(self.position())
    return natoms

  def energy(self): # eV
    energy = self.traj[self.step].get_total_energy()
  #   energy  = ! grep '!' $self.file | tail -1
  #   energy = float(energy[0].split('=')[-1].replace(' Ry', ''))
    return energy

  def force(self): # eV/ A
    force = self.traj[self.step].get_forces()
  #   d = ! grep -A 6 'Forces acting on atoms ' $self.file | tail -$(($self.natoms + 2))
  #   d = d[1:-1]
  #   force = np.array([[float(num) for num in  i.split('=')[-1].split(' ') if num != ''] for i in d])
    return force

  def position(self): # A
    position = self.traj[self.step].get_positions()
    return position

  def atomic_numbers(self):
    atomic_numbers = self.traj[self.step].get_atomic_numbers()
    return atomic_numbers

  def cell(self): # A
    cell = np.array(self.traj[self.step].get_cell())
    return cell

  def Chemical_composition(self):
    Chemical_composition = self.traj[self.step].symbols
    return Chemical_composition


################################## from traj to json
class traj_to_json:
  def __init__(self, pos, lattice, atomic_numbers, natoms, E, forces, key, pos_unit = 'cartesian', length_unit= 'angstrom'):
    self.pos = pos.tolist()
    self.lattice = lattice.tolist()
    self.atomic_numbers = atomic_numbers.tolist()
    self.natoms = natoms
    self.E = E
    self.forces = forces.tolist()
    self.key = key
    self.pos_unit = pos_unit
    self.length_unit = length_unit

  def build_atoms(self):
    atoms = []
    labels = list(range(self.natoms))
    system_atomic_symbols = [ptable[int(atom)] for atom in self.atomic_numbers]
    for idx, (atom, pos, force) in enumerate(zip(system_atomic_symbols, self.pos, self.forces)):
      species = atom
      atoms.append([labels[idx],species, pos, force])

    return atoms

  def dump_json(self):
    json_dict = {}
    json_dict["key"] = self.key
    json_dict["atomic_position_unit"] = self.pos_unit
    json_dict["unit_of_length"] = self.length_unit
    json_dict["energy"] = [self.E, "eV"]
    json_dict["lattice_vectors"] = self.lattice
    json_dict["atoms"] = self.build_atoms()

    with open(f'{self.key}.example', 'w') as json_file:
         json.dump(json_dict, json_file)
         print('Done')


def read_pwo():
     energy  = ! grep '!' $self.file | tail -1
     energy = float(energy[0].split('=')[-1].replace(' Ry', ''))
     d = ! grep -A 6 'Forces acting on atoms ' $self.file | tail -$(($self.natoms + 2))
     d = d[1:-1]
     force = np.array([[float(num) for num in  i.split('=')[-1].split(' ') if num != ''] for i in d])

