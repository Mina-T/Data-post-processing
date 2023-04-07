import numpy as np

# with open('energies.txt') as file:
#     lines = file.readlines()
#     # print(lines[0])
#     lines= [line.split() for line in lines[1:]]
#     for line in lines:
#         for idx, item in enumerate(line):
#             line[idx] = float(item)
      
# All_structures = lines # a list of lists


All_G = np.load('All_G.npy')
All_G_size = range(All_G.shape[0])
structures = [All_G[i] for i in All_G_size]

# for structure in structures:
#     min_G=100
#     # print(len(structure))
#     i=0
#     while i < len(structure)+1:
#         if structure[i][2] < min_G:
#             min_G == structure[i][2]
# for structure in zip(*structures):
#     print(structure)



for a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,\
    a17,a18,a19,a20,a21,a22 in zip(structures[0],structures[1],\
        structures[2], structures[3], structures[4],structures[5],\
        structures[6],structures[7],structures[8],structures[9],\
            structures[10],structures[11],structures[12],structures[13],\
                structures[14],structures[15],structures[16],\
            structures[17],structures[18],structures[19],structures[20],\
                structures[21],structures[22]):
        
        # all_a = [a0[2],a1[2],a2[2],a3[2],a4[2],a5[2],a6[2],a7[2],a8[2],\
                # a9[2],a10[2],a11[2],a12[2],a13[2],a14[2],a15[2],a16[2],\
                # a17[2],a18[2],a19[2],a20[2],a21[2],a22[2]]
        counter = 0
        all_a = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,\
        a17,a18,a19,a20,a21,a22]
        min_E = np.amin([a[2] for a in all_a])
        print('min_E: ', min_E, 'index: ' , np.argmin([a[2] for a in all_a]))
        a_min = all_a[np.argmin([a[2] for a in all_a])]
        print(a_min)
        # file = open('G{0}.dat'.format(np.argmin([a[2] for a in all_a])), 'ab')
        # np.savetxt(file, a_min ,fmt='%.6f')
        print(counter)
        counter += 1

        
        
        
        
        
        # all_a = ['a{0}'.format(i) for i in range(23)]
        # print(a0)        
                # with open('G{0}'.format(np.argmin(all_a)), 'a+') as file:
        #     file.write(str(all_a[np.argmin(all_a)]))
