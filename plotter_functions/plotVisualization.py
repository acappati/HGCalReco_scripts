"""
do visualization of the data
"""

import os
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

# import modules

plt.style.use(hep.style.CMS)
mpl.use('agg')


# function to plot visualization plots
def doVisualizationPlots(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str) -> None:

    # PIONS
    shower_en_pi = []
    shower_eta_pi = []
    # read 3D objects
    # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
    for i_file_pi in data_list_pi:
        for i_evt_pi in i_file_pi:
            trackster_pi = i_evt_pi.clus3d_feat.numpy()

            shower_en_pi.append(trackster_pi[2])
            shower_eta_pi.append(trackster_pi[0])

    # PHOTONS
    shower_en_pho = []
    shower_eta_pho = []
    # read 3D objects
    # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
    for i_file_pho in data_list_pho:
        for i_evt_pho in i_file_pho:
            trackster_pho = i_evt_pho.clus3d_feat.numpy()

            shower_en_pho.append(trackster_pho[2])
            shower_eta_pho.append(trackster_pho[0])

    # plot eta vs energy
    fig, axs = plt.subplots(1, 2, figsize=(20,10), tight_layout=True)
    axs[0].hist2d(shower_en_pho, shower_eta_pho, bins=50, range=((0,1200),(1.2,3.2)), density=True, cmap='Oranges')
    axs[0].set_xlabel('Energy')
    axs[0].set_ylabel('eta')
    axs[0].set_title(r'$\gamma$')
    axs[1].hist2d(shower_en_pi, shower_eta_pi, bins=50, range=((0,1200),(1.2,3.2)), density=True, cmap='Greens')
    axs[1].set_xlabel('Energy')
    axs[1].set_ylabel('eta')
    axs[1].set_title(r'$\pi$')
    plt.savefig(os.path.join(out_dir, 'eta_en_trackster.png')) #save plot
    plt.close(fig)
