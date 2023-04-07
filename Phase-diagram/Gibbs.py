#
import numpy as np

with open('energies.txt') as file:
    lines = file.readlines()
    print(lines[0])
    lines= [line.split() for line in lines[1:]]
    for line in lines:
        for idx, item in enumerate(line):
            line[idx] = float(item)
      
All_structures = lines 
number_of_structures = len(All_structures)

##
eV = 13.605698066
E_CO = -60.0141661681 # Ry
E_O2 = -83.0451292300 # Ry
E_TiO2_Pt6 = -7757.0956273604
   

M = np.arange(-4, 0.0, 0.004, dtype = float)
S = M

x = len(M)*len(S)


# All_G = np.empty(shape=(0,3),dtype=list)
All_G = []
class Gibss_free_energy:
    def __init__(self, E_tot, n_Pt, n_O, n_CO):
        self.E_tot = E_tot
        self.n_O = n_O
        self.n_CO = n_CO
        self.n_Pt = n_Pt
    
    def calculate_energy(self):
        # arr = np.empty(shape=(23, 1000001,3))
        # arr[idx][0] = idx
        # print(arr)
        arr = []
        for m in M:
            for s in S:   
                G = ((((self.E_tot-E_TiO2_Pt6)-self.n_O*0.5*E_O2)-self.n_CO*E_CO)*eV-self.n_O*m-self.n_CO*s)
                arr.append([m, s, G])        
        All_G.append(np.array(arr))
        print(All_G)

      
for structure in All_structures: 
        idx = All_structures.index(structure)
        print(idx)
        E_tot = structure[0]
        n_Pt = structure[1]
        n_O = structure[2]
        n_CO = structure[3]
        G_final = Gibss_free_energy(E_tot, n_Pt, n_O, n_CO)
        G_final.calculate_energy()
        
print(len(All_G))       
All_G = np.array(All_G)
print(All_G.shape)
np.save('All_G', All_G)

