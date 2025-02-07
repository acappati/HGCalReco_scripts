"""
function to plot gen particle features
"""

import os
import string
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

# import modules

plt.style.use(hep.style.CMS)
mpl.use('agg')


def dogunmatrix(data_list: List[Data]) -> Tensor:
    '''
    function that reads data and returns the matrix
    '''

    gun_matrix = []
    for i_file in data_list:
        for i_evt in i_file:
            gun_matrix.append(i_evt.gun_feat.numpy())

    gun_matrix = np.vstack(gun_matrix)

    return gun_matrix



def doGunPlots(data_dict: Dict[str, List[Data]],
               out_dir: str,
               norm: bool = True) -> None:
    '''
    function to plot features of the gun
    trkguneta,trkgunphi,trkgunen : Gun properties eta,phi,energy
    '''

    # take matrix of gun properties
    matrix_dict = {}
    for key in data_dict:
        gun_matrix = dogunmatrix(data_dict[key])
        matrix_dict[key] = gun_matrix

        print(key, gun_matrix.shape)

    print(matrix_dict.keys())


    if norm:
        nameAppendix = '_w_norm'
    else:
        nameAppendix = '_wo_norm'


    # --- plot gun features

    # plot total energy
    fig, axs = plt.subplots(figsize=(8,6), layout="constrained")

    for key, value in matrix_dict.items():
        axs.hist(value[:,3], bins=120, range=(0.,1200.),
                 density=norm, histtype='step', linewidth=2, label=key)
    axs.legend()
    axs.set_xlabel('total energy')
    axs.set_ylabel('# trk')
    axs.set_yscale('log')

    plotname = 'gun_energy' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname)) #save plot
    plt.close(fig)


    # plot eta
    fig, axs = plt.subplots(figsize=(8,6), layout="constrained")

    for key, value in matrix_dict.items():
        axs.hist(value[:,0], bins=50, range=(1.2,3.2),
                 density=norm, histtype='step', linewidth=2, label=key)
    axs.legend()
    axs.set_xlabel('eta')
    axs.set_ylabel('# trk')

    plotname = 'gun_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname)) #save plot
    plt.close(fig)


    # # plot phi
    fig, axs = plt.subplots(figsize=(8,6), layout="constrained")

    for key, value in matrix_dict.items():
        axs.hist(value[:,1], bins=50, range=(-4.,4.),
                 density=norm, histtype='step', linewidth=2, label=key)
    axs.legend()
    axs.set_xlabel('phi')
    axs.set_ylabel('# trk')

    plotname = 'gun_phi' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname)) #save plot
    plt.close(fig)
