import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

P_ff = [0,   0,     0.0983, 0.0619, 0,   0,     0.0512, 0.0196]
P_fb = [0.1, 0.085, 0,      0,      0.1, 0.085, 0.1,    0.085 ]

N_ff = 902
N_fb = 1200

nu_ff = 15 # 15
nu_fb = 5  # 5

L23, L4, L5, L6 = 0, 1, 1, 1

# 3 conditions: 1) feedforward to L4 and L6 2) feedback to L5 and L6 3) feedback to L23

I_1 = [0,              0,                0.0983*902*15*J_E, 0.0619*902*15*J_E, 0,              0,                0.0512*902*15*J_E, 0.0196*902*15*J_E]
I_2 = [0,              0,                0.0983*902*15*J_E, 0.0619*902*15*J_E, 0.1*1200*5*J_E, 0.085*1200*5*J_E, 0.0512*902*15*J_E + 0.1*1200*5*J_E, 0.0196*902*15*J_E + 0.085*1200*5*J_E]
I_3 = [0.1*1200*5*J_E, 0.085*1200*5*J_E, 0.0983*902*15*J_E, 0.0619*902*15*J_E, 0,              0,                0.0512*902*15*J_E, 0.0196*902*15*J_E]
I_4 = [0.1*1200*5*J_E, 0.085*1200*5*J_E, 0.0983*902*15*J_E, 0.0619*902*15*J_E, 0.1*1200*5*J_E, 0.085*1200*5*J_E, 0.0512*902*15*J_E + 0.1*1200*5*J_E, 0.0196*902*15*J_E + 0.085*1200*5*J_E]

I_all = np.array([I_1, I_2, I_3, I_4])

E = {'K': 12, 'area': 'V1', 'T': 50, 'onset': 10, 'offset': 20}   # experimental parameters

num_conditions = 4
conditions = ['Granular', 'Granular + Deep', 'Granular + Superficial', 'Granular + Superficial + Deep']

peaks = np.zeros((num_conditions, E['K']))

for i in range(num_conditions):
    theta = {
            'I_L23E': I_all[i][0],
            'I_L23I': I_all[i][1],
            'I_L4E':  I_all[i][2],
            'I_L4I':  I_all[i][3],
            'I_L5E':  I_all[i][4],
            'I_L5I':  I_all[i][5],
            'I_L6E':  I_all[i][6],
            'I_L6I':  I_all[i][7],

            'lam_E': 1,
            'lam_I': 0
            }

    Psi, X, S, F_l, F_k, B_k, B_v, Y = F(E, theta, mode='full')  # forward model

    # cast to float32
    B_k = B_k.astype(np.float32)
    peaks[i,:] = np.max(B_k, axis=0)


colors = plt.cm.viridis(np.linspace(0, 1, num_conditions))
plt.figure(figsize=(4, 4))
for i in range(num_conditions):
    plt.plot(np.arange(1, E['K']+1), peaks[i,:], label=conditions[i], color=colors[i], linewidth=2)
# plt.legend()
plt.xlabel('Cortical Depth', fontsize=20)
plt.ylabel('Signal Change (%)', fontsize=20)
plt.xticks([])
plt.yticks(np.linspace(0.1, .7, 3), np.linspace(0.1, .7, 3), fontsize=16)
plt.xlim([1, E['K']])
plt.tight_layout()
plt.savefig('pdf/peaks_ff_fb.pdf', dpi=1200)
plt.show()

import IPython; IPython.embed()






# plt.figure(figsize=(5, 5))
# plt.subplot(211)
# plt.imshow(B_k.T, aspect='auto', cmap='Reds', interpolation='none')
# plt.title('BOLD Signal before downsampling (K x T)')
# plt.colorbar()
# plt.ylabel('Cortical Depth (K)')
# plt.yticks(np.arange(1, E['K'], 3), np.arange(1, E['K'], 3))
# plt.xticks([])
# plt.subplot(212)
# plt.imshow(B_v.T, aspect='auto', cmap='Reds', interpolation='none')
# plt.title('BOLD Signal after downsampling (V x T)')
# plt.colorbar()
# plt.xlabel('Time (s)')
# plt.yticks(np.arange(3), ['Superficial', 'Granular', 'Deep'])
# plt.xticks(np.linspace(0, int((E['T']-1)/0.001), 3), np.linspace(0, E['T'], 3))
# plt.tight_layout()
# plt.savefig('png/BOLD_ff.png', dpi=1200)
# plt.show()
