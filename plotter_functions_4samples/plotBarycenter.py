"""
functions to plot the barycenter, from the training samples
"""


import os
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
from plotter_functions_4samples import _divide_en_categ, _divide_eta_categ

plt.style.use(hep.style.CMS)
mpl.use('agg')



def doBarycenter(data_dict: Dict[str, List[Data]],
                 out_dir: str,
                 norm: bool =True,
                 n_eta_cat: int = 8,
                 n_en_cat: int = 9
                 ):
    """
    function to do plot the barycenter from the training samples
    """


    if norm:
        nameAppendix = '_w_norm'
    else:
        nameAppendix = '_wo_norm'


    ## define disctionaries
    bar_dict = {k: [] for k in data_dict.keys()}
    # dictionaries in eta bins
    bar_eta_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_eta_cat)]
    # dictionaries in en bins
    bar_en_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_en_cat)]


    # define array of strings for eta bins
    eta_bin_str = [
        '1.65 < eta < 1.75',
        '1.75 < eta < 1.85',
        '1.85 < eta < 1.95',
        '1.95 < eta < 2.05',
        '2.05 < eta < 2.15',
        '2.15 < eta < 2.35',
        '2.35 < eta < 2.55',
        '2.55 < eta < 2.75'
        ]
    # define array of strings for energy bins
    en_bin_str = [
        '0 < E < 100 GeV',
        '100 < E < 200 GeV',
        '200 < E < 300 GeV',
        '300 < E < 400 GeV',
        '400 < E < 500 GeV',
        '500 < E < 600 GeV',
        '600 < E < 700 GeV',
        '700 < E < 800 GeV',
        'E > 800 GeV'
        ]



    for key, value in data_dict.items():
        # loop over files in data dict, for a certain key
        for i_file in value:
            # loop over all events in one file
            for i_evt in i_file:

                #2D object
                layerClusterMatrix = i_evt.clus2d_feat.numpy()

                #3D object
                trackster = i_evt.clus3d_feat.numpy()


                ## compute the barycenter
                # mean of the z coordinate, weighted by the energy
                barycenter_z = np.average(layerClusterMatrix[:,2], weights=layerClusterMatrix[:,3])


                # fill inclusive dictionary
                bar_dict[key].append(barycenter_z)


                # get the eta category number
                cat_eta_n = _divide_eta_categ(trackster[0])
                # check if values outside eta categories
                if cat_eta_n < 0: continue
                # divide in eta bins
                bar_eta_dict[cat_eta_n][key].append(barycenter_z)


                # get the energy category number
                cat_en_n = _divide_en_categ(trackster[2])
                # divide in energy bins
                bar_en_dict[cat_en_n][key].append(barycenter_z)



    ## ----------- plot inclusive
    fig, axs = plt.subplots(1, 1, figsize=(20, 10), dpi=80, tight_layout=True)

    # hist of min_clusL
    for key, value in bar_dict.items():
        axs.hist(value,
                    bins=50,
                    range=(300, 550),
                    density=norm,
                    histtype='step',
                    linewidth=2,
                    label=key)
    axs.legend()
    axs.set_xlabel('Barycenter')
    axs.set_ylabel('# events')

    plotname = 'barycenter' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig)



    ## ----------- plot in cat of eta
    # boundary between low and high density: 2.15 eta
    fig, axs = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs.flatten()

    for cat in range(n_eta_cat):
        for key in bar_eta_dict[cat]:
            axs.flatten()[cat].hist(bar_eta_dict[cat][key],
                                     bins=50,
                                     range=(300, 550),
                                     density=norm,
                                     histtype='step',
                                     linewidth=2,
                                     label=key)
        axs.flatten()[cat].legend()
        axs.flatten()[cat].set_xlabel('Barycenter')
        axs.flatten()[cat].set_ylabel('# events')
        # add a box containing the eta range
        axs.flatten()[cat].text(
            0.7,
            0.2,
            eta_bin_str[cat],
            transform=axs.flatten()[cat].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'barycenter_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig)



    ## ----------- plot in cat of energy
    # boundary every 100 GeV
    fig, axs = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs.flatten()

    for cat in range(n_en_cat):
        for key in bar_en_dict[cat]:
            axs.flatten()[cat].hist(bar_en_dict[cat][key],
                                     bins=50,
                                     range=(300, 550),
                                     density=norm,
                                     histtype='step',
                                     linewidth=2,
                                     label=key)
        axs.flatten()[cat].legend()
        axs.flatten()[cat].set_xlabel('Barycenter')
        axs.flatten()[cat].set_ylabel('# events')
        # add a box containing the energy range
        axs.flatten()[cat].text(
            0.6,
            0.3,
            en_bin_str[cat],
            transform=axs.flatten()[cat].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'barycenter_en' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig)
