"""
function to plot gen particle features
"""

import os
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

# import modules

plt.style.use(hep.style.CMS)
mpl.use('agg')


# function to plot features of the gun
# trkguneta,trkgunphi,trkgunen : Gun properties eta,phi,energy
def doGunPlots(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str) -> None:

    # pions
    gun_matrix_pi = []
    for i_file in data_list_pi:
        for i_evt in i_file:
            gun_matrix_pi.append(i_evt.gun_feat.numpy())

    gun_matrix_pi = np.vstack(gun_matrix_pi)
    print(gun_matrix_pi.shape)

    # photons
    gun_matrix_pho = []
    for i_file in data_list_pho:
        for i_evt in i_file:
            gun_matrix_pho.append(i_evt.gun_feat.numpy())

    gun_matrix_pho = np.vstack(gun_matrix_pho)
    print(gun_matrix_pho.shape)

    # plot gun features
    fig, axs = plt.subplots(3, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs.flatten()
    # plot eta
    axs.flatten()[0].hist(gun_matrix_pi[:,0], bins=50, range=(1.2,3.2),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[0].hist(gun_matrix_pho[:,0], bins=50, range=(1.2,3.2),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[0].legend()
    axs.flatten()[0].set_xlabel('eta')
    axs.flatten()[0].set_ylabel('# trk')
    # plot phi
    axs.flatten()[1].hist(gun_matrix_pi[:,1], bins=50, range=(-4.,4.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[1].hist(gun_matrix_pho[:,1], bins=50, range=(-4.,4.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[1].legend()
    axs.flatten()[1].set_xlabel('phi')
    axs.flatten()[1].set_ylabel('# trk')
    # plot energy in EM part of HGCal
    axs.flatten()[2].hist(gun_matrix_pi[:,2], bins=50, range=(0.,1200.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[2].hist(gun_matrix_pho[:,2], bins=50, range=(0.,1200.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[2].legend()
    axs.flatten()[2].set_xlabel('energy in CEE')
    axs.flatten()[2].set_ylabel('# trk')
    # plot total energy
    axs.flatten()[3].hist(gun_matrix_pi[:,3], bins=50, range=(0.,1200.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[3].hist(gun_matrix_pho[:,3], bins=50, range=(0.,1200.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[3].legend()
    axs.flatten()[3].set_xlabel('total energy')
    axs.flatten()[3].set_ylabel('# trk')
    # plot ratio
    axs.flatten()[4].hist(gun_matrix_pi[:,4], bins=50, range=(0.,5.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[4].hist(gun_matrix_pho[:,4], bins=50, range=(0.,5.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[4].legend()
    axs.flatten()[4].set_xlabel('ratio')
    axs.flatten()[4].set_ylabel('# trk')


    plt.savefig(os.path.join(out_dir, 'gunFeats.png')) #save plot
    plt.close(fig)
