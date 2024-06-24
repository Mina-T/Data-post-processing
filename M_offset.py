import os
import ase
from ase.io import extxyz
import numpy as np

#all elements
#el = 'H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr'
#elements = el.split(', ')

#offsets = {}
#for el in elements:
#    offsets[el] = 0.0
#offsets['N']= -120.2748
#offsets.values()

#*******************************************************************************************************************
el = 'Ag,Al,As,Au,B,Bi,C,Ca,Cd,Cl,Co,Cr,Cs,Cu,Fe,Ga,Ge,H,Hf,Hg,In,Ir,K,Mn,Mo,N,Na,Nb,Ni,O,Os,P,Pb,Pd,Pt,Rb,Re,Rh,Ru,S,Sb,Sc,Se,Si,Sn,Sr,Ta,Tc,Te,Ti,Tl,V,W,Y,Zn,Zr'
elements = el.split(',')
offset = [-1.89,-3.62,-4.49,-2.69,-6.08,-3.33,-8.29,-2.44,-0.21,-3.08,-6.23,-8.44,-1.50,-2.96,-7.18,-2.57,-4.07,-3.40,-9.92,0.29,-2.06,-8.29,-1.53,-7.84,-9.83,-8.42,-1.54,-9.59,-4.89,-4.92,-10.13,-5.29,-3.02,-4.79,-5.78,-1.35,-11.05,-6.79,-8.38,-4.61,-3.65,-6.73,-3.81,-5.31,-3.38,-2.35,-11.14,-9.27,-3.03,-7.75,-1.81,-8.38,-11.67,-7.03,-0.67,-8.54]

offsets = {}
for el, off in zip(elements, offset):
    offsets[el] = off


ptable = ['null', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo']

def calc_offsetE(atomic_numbers):
    offsetE = 0
    for num in atomic_numbers:
        element = ptable[num]
        offsetE += offsets[element]
    return offsetE

def remove_offset(path, n_files, n_start = 0):
    os.chdir(path)
    n_structures = 0
    bonding_effect_atm = 0
    for i in range(n_start , n_files):
        print(f'file {i}.extxyz', flush = True)
        idx = 0
        File = open(f'{i}.extxyz', 'r')
        txt_file = open(f'{i}.txt')
        read = [line for line in txt_file]
        Id_list = [line.split(',')[0].replace('random', '') for line in read]
        dataset = np.array([], dtype=object)
        dataset = np.hstack((dataset, np.array(['index = 0', 'Number of atoms = 1', 'atomic numbers = 2',
        'Chemical composition = 3', 'Energy = 4', 'bonding effect = 5', 'Id = 6'])))
        
        while True:
                try:
                    structure = list(extxyz.read_xyz(File, index = idx, properties_parser = extxyz.key_val_str_to_dict))
                    symbol = str(structure[0].symbols)
                    atomic_symbols = np.array(structure[0].get_chemical_symbols())
                    atomic_numbers = np.array( [ptable.index(i) for i in atomic_symbols])
                    natoms = len(atomic_numbers)
                    energy = structure[0].get_total_energy()
                    offset_E = calc_offsetE(atomic_numbers)
                    bonding_effect = energy - offset_E
                    bonding_effect_atm += (bonding_effect/natoms)
                    Id = str(i)+'_'+str(idx)+'_'+Id_list[idx]
                    lst = [idx, natoms, atomic_numbers, symbol, energy, bonding_effect, Id]
                    dataset = np.vstack((dataset, np.array(lst, dtype=object)))
                    idx +=1
                    n_structures += 1
        
                except:
                    write = open(f'offset_removed_{i}.npy', 'wb')
                    np.save(write, dataset)
                    break
    print(f'For {n_files} files and {n_structures} structures, the average bonding effect is {bonding_effect_atm/n_structures}')

