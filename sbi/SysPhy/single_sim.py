import numpy as np
import pylab as plt

from worker import F

J_E = 87.8e-3

theta = {'a': 48, 'b': 981, 'd': 8.9e-3, 'tau_m': 10e-3, 'tau_s': .5e-3, 'C_m': 250e-6,                                 # intrinsic neuronal parameters
            'I_L4E': 0.0983*902*18*J_E, 'I_L4I': 0.0619*902*18*J_E, 'I_L6E': 0.0512*902*20*J_E, 'I_L6I': 0.0196*902*20*J_E,# external input
            'lam_E': 1, 'lam_I': 0, 'c1': 0.6, 'c2': 1.5, 'c3': 0.6,                                                       # neurovascular coupling parameters
            'E_0v': 0.35, 'V_0t': 2, 'TE': 0.028}                                                                          # hemodynamic parameters

E = {'K': 22, 'area': 'V1', 'T': 50, 'onset': 10, 'offset': 20}   # experimental parameters

Psi, X, S, F_l, F_k, B_k, B_v = F(E, theta, test=True)  # forward model

# casting to float32
X = X.astype(np.float32)
S = S.astype(np.float32)
F_l = F_l.astype(np.float32)
F_k = F_k.astype(np.float32)
B_k = B_k.astype(np.float32)
B_v = B_v.astype(np.float32)

fig = plt.figure(figsize=(7, 7))
plt.subplot(5, 1, 1)
plt.title(r'Layer specific neural activity ($S$)')
plt.imshow(S.T, aspect='auto', cmap='Reds', interpolation='none')
plt.yticks(np.arange(4), ['L23', 'L4', 'L5', 'L6'])
plt.colorbar()
plt.subplot(5, 1, 2)
plt.title(r'Cerebral Blood Flow before upsampling ($F_l$)')
plt.imshow(F_l.T, aspect='auto', cmap='Reds', interpolation='none')
plt.yticks(np.arange(4), ['L23', 'L4', 'L5', 'L6'])
plt.colorbar()
plt.subplot(5, 1, 3)
plt.title(r'Cerebral Blood Flow after upsampling ($F_k$)')
plt.imshow(F_k.T, aspect='auto', cmap='Reds', interpolation='none')
plt.colorbar()
plt.subplot(5, 1, 4)
plt.title(r'BOLD signal before downsampling ($B_k$)')
plt.imshow(B_k.T, aspect='auto', cmap='Reds', interpolation='none')
fig.text(0.07, 0.4, 'Cortical depth (K)', va='center', rotation='vertical')
plt.colorbar()
plt.subplot(5, 1, 5)
plt.title(r'BOLD signal after downsampling ($B_v$)')
plt.imshow(B_v.T, aspect='auto', cmap='Reds', interpolation='none')
plt.yticks(np.arange(3), ['Superficial', 'Granular', 'Deep'])
plt.colorbar()
plt.tight_layout(pad=1)
plt.savefig('pdf/ff_L4.pdf', format='pdf', dpi=1200)

# plot Psi observables
plt.figure(figsize=(7, 7))
plt.subplot(5, 1, 1)
plt.title(r'Peak position ($\Psi_{peak\_pos}$)')
plt.plot(Psi['peak_pos'])
plt.subplot(5, 1, 2)
plt.title(r'Peak amplitude ($\Psi_{peak\_amp}$)')
plt.plot(Psi['peak_amp'])
plt.subplot(5, 1, 3)
plt.title(r'Area under the curve ($\Psi_{area}$)')
plt.plot(Psi['area'])
plt.subplot(5, 1, 4)
plt.title(r'Upslope ($\Psi_{upslope}$)')
plt.plot(Psi['upslope'])
plt.subplot(5, 1, 5)
plt.title(r'Downslope ($\Psi_{downslope}$)')
plt.plot(Psi['downslope'])
plt.tight_layout(pad=1)
plt.savefig('pdf/ff_L4_Psi.pdf', format='pdf', dpi=1200)

plt.figure(figsize=(7, 7))
peaks_S = np.max(S, axis=0)
plt.barh(width=peaks_S, y=np.arange(4), linewidth=2, color='black')
plt.xticks([])
plt.yticks([])
plt.gca().invert_yaxis()
plt.axis('off')
plt.savefig('pdf/ff_L4_S.pdf', format='pdf', dpi=1200)

plt.figure(figsize=(7, 7))
peaks_F_k = np.max(F_k, axis=0)
plt.plot(peaks_F_k, np.arange(1, E['K']+1), linewidth=2, color='black')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('pdf/ff_L4_F_k.pdf', format='pdf', dpi=1200)

plt.figure(figsize=(7, 7))
peaks = np.max(B_k, axis=0)
plt.plot(peaks, np.arange(1, E['K']+1), linewidth=2, color='black')
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.savefig('pdf/ff_L4_B_k.pdf', format='pdf', dpi=1200)
