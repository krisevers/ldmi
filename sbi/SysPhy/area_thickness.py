import numpy as np
import pylab as plt

from worker import F
from utils import get_L2K, get_N

"""
Exploration of area specific neuronal densities and thicknesses on the laminar BOLD signal.
"""

areas = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd', 'PIP', 'PO', 'DP', 'MIP', 'MDP', 
         'VIP', 'LIP', 'PITv', 'PITd', 'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp', 'STPa', '46', 'AITd']

K = 12
L = 4

F_max = np.zeros((K, len(areas)))
B_max = np.zeros((K, len(areas)))

for i, area in enumerate(areas):
    print('Area: {} | {}/{}'.format(area, i+1, len(areas)), end='\r')

    E = {'K': 12, 'area': area, 'T': 30, 'onset': 5, 'offset': 10}   # experimental parameters

    _, _, _, _, F_k, B_k, _ = F(E, test=True)  # forward model

    F_max[:, i] = np.max(F_k, axis=0) # maximum of each layer
    B_max[:, i] = np.max(B_k, axis=0) # maximum of each layer


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(F_max, aspect='auto')
plt.title('F')
plt.colorbar()
plt.xlabel('Area')
plt.ylabel('Cortical Depth (K)')
plt.xticks(np.arange(len(areas)), areas, rotation=90)
plt.yticks(np.arange(K), np.arange(K))
plt.subplot(122)
plt.imshow(B_max, aspect='auto')
plt.title('B')
plt.colorbar()
plt.xlabel('Area')
plt.ylabel('Cortical Depth (K)')
plt.xticks(np.arange(len(areas)), areas, rotation=90)

plt.tight_layout()
plt.savefig('pdf/area_thickness.pdf', dpi=300)
plt.show()


import IPython; IPython.embed() 