"""
function to plot the fraction of energy in the CEH
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.signal import argrelmin, find_peaks
from scipy.stats import gaussian_kde
#import pandas as pd
from tqdm import tqdm as tqdm

plt.style.use(hep.style.CMS)
mpl.use('agg')


def plotFractionCEH(data_list_pho, data_list_pi, out_dir) -> None:
    """
    function to plot the fraction of energy in the CEH
    """

    # create arrays for fraction of energy in CEH per event
    fracH_pho = []
    fracH_pi = []


    ### PHOTONS
    # loop over files in data list
    for i_file in data_list_pho:
        # loop over all events in one file
        for i_evt in i_file:

            # read 2D objects
            layerClusterMatrix = i_evt.clus2d_feat.numpy()

            ## compute the energy fraction in the hadronic part of HGCal
            # consider the last 22 layers
            fracH_pho.append(np.sum(layerClusterMatrix[layerClusterMatrix[:,5]> 26][:,3])/np.sum(layerClusterMatrix[:,3]))


    ### PIONS
    # loop over files in data list
    for i_file in data_list_pi:
        # loop over all events in one file
        for i_evt in i_file:

            # read 2D objects
            layerClusterMatrix = i_evt.clus2d_feat.numpy()

            ## compute the energy fraction in the hadronic part of HGCal
            # consider the last 22 layers
            fracH_pi.append(np.sum(layerClusterMatrix[layerClusterMatrix[:,5]> 26][:,3])/np.sum(layerClusterMatrix[:,3]))


    # plot energy fraction in the hadronic part
    fig1, ax1 = plt.subplots(figsize=(12,8), dpi=80, tight_layout=True)
    ax1.hist(fracH_pi, bins=100, range=(0.,1.), density=True, color='green', alpha=0.4, label=r'$\pi$')
    ax1.hist(fracH_pho, bins=100, range=(0.,1.), density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy fraction in CEH')
    ax1.set_ylabel('# events')

    plt.savefig(os.path.join(out_dir, 'en_fracH.png')) #save plot
    plt.close(fig1)
