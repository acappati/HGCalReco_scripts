"""
functions to do histograms from the training samples
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




def doHisto(data_dict: Dict[str, List[Data]],
            out_dir: str,
            norm: bool =True,
            n_eta_cat: int = 8,
            n_en_cat: int = 9
            ):
    """
    function to do histograms from the training samples
    """

    if norm:
        nameAppendix = '_w_norm'
    else:
        nameAppendix = '_wo_norm'

    ### PHOTONS
    min_clusL_dict = {k: [] for k in data_dict.keys()} # dict of all min_clusL
    max_clusL_dict = {k: [] for k in data_dict.keys()} # dict of all max_clusL
    extShower_dict = {k: [] for k in data_dict.keys()} # dict for shower ext
    showerEn_dict  = {k: [] for k in data_dict.keys()} # dict for energy
    showerEta_dict = {k: [] for k in data_dict.keys()} # dict for eta

    # define dictionaries of min_clusL and max_clusL for each eta bin
    min_clusL_cat_eta_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_eta_cat)]
    max_clusL_cat_eta_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_eta_cat)]
    extShower_cat_eta_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_eta_cat)]
    showerEn_cat_eta_dict  = [{k: [] for k in data_dict.keys()} for _ in range(n_eta_cat)]
    showerEta_cat_eta_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_eta_cat)]

    # define dictionaries of min_clusL and max_clusL for each pT bin
    min_clusL_cat_en_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_en_cat)]
    max_clusL_cat_en_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_en_cat)]
    extShower_cat_en_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_en_cat)]
    showerEn_cat_en_dict  = [{k: [] for k in data_dict.keys()} for _ in range(n_en_cat)]
    showerEta_cat_en_dict = [{k: [] for k in data_dict.keys()} for _ in range(n_en_cat)]

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

                # --- read 3D objects
                # the clus3d_feat is a tensor of only 6 features:
                # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
                trackster = i_evt.clus3d_feat.numpy()  # transform the tensor in numpy array

                # fill array of all min and max cluster Layer
                min_clusL_dict[key].append(trackster[4])
                max_clusL_dict[key].append(trackster[5])
                extShower_dict[key].append(abs(trackster[5] - trackster[4]))
                showerEn_dict[key].append(trackster[2])
                showerEta_dict[key].append(trackster[0])

                # get the eta category number
                cat_eta_n = _divide_eta_categ(trackster[0])
                # check if values outside eta categories
                if cat_eta_n < 0: continue
                # divide in eta bins
                min_clusL_cat_eta_dict[cat_eta_n][key].append(trackster[4])
                max_clusL_cat_eta_dict[cat_eta_n][key].append(trackster[5])
                extShower_cat_eta_dict[cat_eta_n][key].append(abs(trackster[5] - trackster[4]))
                showerEn_cat_eta_dict[cat_eta_n][key].append(trackster[2])
                showerEta_cat_eta_dict[cat_eta_n][key].append(trackster[0])

                # get the energy category number
                cat_en_n = _divide_en_categ(trackster[2])
                # divide in energy bins
                min_clusL_cat_en_dict[cat_en_n][key].append(trackster[4])
                max_clusL_cat_en_dict[cat_en_n][key].append(trackster[5])
                extShower_cat_en_dict[cat_en_n][key].append(abs(trackster[5] - trackster[4]))
                showerEn_cat_en_dict[cat_en_n][key].append(trackster[2])
                showerEta_cat_en_dict[cat_en_n][key].append(trackster[0])



    # histos min-max L - inclusive
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=80, tight_layout=True)
    #binEdges_list = np.arange(0, 50, 1)  # to contain all 48 layers

    # hist of min_clusL
    for key, value in min_clusL_dict.items():
        axs[0].hist(value, bins=50, range=(0, 50),
                    density=norm, histtype='step', label=key)
    axs[0].legend()
    axs[0].set_xlabel('min clusL')
    axs[0].set_ylabel('# trk')

    # hist of max_clusL
    for key, value in max_clusL_dict.items():
        axs[1].hist(value, bins=50, range=(0, 50),
                    density=norm, histtype='step', label=key)
    axs[1].legend()
    axs[1].set_xlabel('max clusL')
    axs[1].set_ylabel('# trk')

    plotname = 'minmaxL' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig)

    # hist of shower extension - inclusive
    fig0, axs0 = plt.subplots(1, 1, figsize=(20,10), dpi=80, tight_layout=True)
    for key, value in extShower_dict.items():
        axs0.hist(value, bins=50, range=(0, 50),
                    density=norm, histtype='step', label=key)
    axs0.legend()
    axs0.set_xlabel('shower extension')
    axs0.set_ylabel('# trk')

    plotname = 'extShower' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig0)


    ### do plots in bins of eta: min_clusL
    # boundary between low and high density: 2.15 eta
    fig1, axs1 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs1.flatten()

    for cat in range(n_eta_cat):
        for key in min_clusL_cat_eta_dict[cat]:
            axs1.flatten()[cat].hist(min_clusL_cat_eta_dict[cat][key],
                                    bins=50,
                                    range=(0, 50),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs1.flatten()[cat].legend()
        axs1.flatten()[cat].set_xlabel('min clusL')
        axs1.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs1.flatten()[cat].text(
            0.7,
            0.6,
            eta_bin_str[cat],
            transform=axs1.flatten()[cat].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'minL_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig1)

    ### do plots in bins of eta: max_clusL
    # boundary between low and high density: 2.15 eta
    fig2, axs2 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs2.flatten()

    for cat in range(n_eta_cat):
        for key in max_clusL_cat_eta_dict[cat]:
            axs2.flatten()[cat].hist(max_clusL_cat_eta_dict[cat][key],
                                    bins=50,
                                    range=(0, 50),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs2.flatten()[cat].legend()
        axs2.flatten()[cat].set_xlabel('max clusL')
        axs2.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs2.flatten()[cat].text(
            0.7,
            0.6,
            eta_bin_str[cat],
            transform=axs2.flatten()[cat].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'maxL_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig2)

    # do plots in bins of eta: shower extension
    # boundary between low and high density: 2.15 eta
    fig3, axs3 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()

    for cat in range(n_eta_cat):
        for key in extShower_cat_eta_dict[cat]:
            axs3.flatten()[cat].hist(extShower_cat_eta_dict[cat][key],
                                    bins=50,
                                    range=(0, 50),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower extension')
        axs3.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs3.flatten()[cat].text(
            0.7,
            0.6,
            eta_bin_str[cat],
            transform=axs3.flatten()[cat].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'extShower_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig3)

    # do plots in bins of eta: shower energy
    # boundary between low and high density: 2.15 eta
    fig3, axs3 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()

    for cat in range(n_eta_cat):
        for key in showerEn_cat_eta_dict[cat]:
            axs3.flatten()[cat].hist(showerEn_cat_eta_dict[cat][key],
                                    bins=55,
                                    range=(0, 1100),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower energy')
        axs3.flatten()[cat].set_ylabel('# trk')
        axs3.flatten()[cat].set_yscale('log')
        # add a box containing the eta range
        axs3.flatten()[cat].text(
            0.7,
            0.6,
            eta_bin_str[cat],
            transform=axs3.flatten()[cat].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'showerEn_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig3)

    # do plots in bins of eta: shower eta
    # boundary between low and high density: 2.15 eta
    fig3, axs3 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()

    for cat in range(n_eta_cat):
        for key in showerEta_cat_eta_dict[cat]:
            axs3.flatten()[cat].hist(showerEta_cat_eta_dict[cat][key],
                                    bins=48,
                                    range=(1.5, 3.1),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower eta')
        axs3.flatten()[cat].set_ylabel('# trk')
        axs3.flatten()[cat].set_yscale('log')
        # add a box containing the eta range
        axs3.flatten()[cat].text(
            0.7,
            0.6,
            eta_bin_str[cat],
            transform=axs3.flatten()[cat].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'showerEta_eta' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig3)

    # do plots in bins of energy: min_clusL
    # boundary every 100 GeV
    fig4, axs4 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs4.flatten()

    for cat in range(n_en_cat):
        for key in min_clusL_cat_en_dict[cat]:
            axs4.flatten()[cat].hist(min_clusL_cat_en_dict[cat][key],
                                    bins=50,
                                    range=(0, 50),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs4.flatten()[cat].legend()
        axs4.flatten()[cat].set_xlabel('min clusL')
        axs4.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs4.flatten()[cat].text(
            0.6,
            0.6,
            en_bin_str[cat],
            transform=axs4.flatten()[cat].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'minL_en' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig4)

    # do plots in bins of energy: max_clusL
    # boundary every 100 GeV
    fig5, axs5 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs5.flatten()

    for cat in range(n_en_cat):
        for key in max_clusL_cat_en_dict[cat]:
            axs5.flatten()[cat].hist(max_clusL_cat_en_dict[cat][key],
                                    bins=50,
                                    range=(0, 50),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs5.flatten()[cat].legend()
        axs5.flatten()[cat].set_xlabel('max clusL')
        axs5.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs5.flatten()[cat].text(
            0.6,
            0.6,
            en_bin_str[cat],
            transform=axs5.flatten()[cat].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'maxL_en' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig5)

    # do plots in bins of energy: shower extension
    # boundary every 100 GeV
    fig6, axs6 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs6.flatten()

    for cat in range(n_en_cat):
        for key in extShower_cat_en_dict[cat]:
            axs6.flatten()[cat].hist(extShower_cat_en_dict[cat][key],
                                    bins=50,
                                    range=(0, 50),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower extension')
        axs6.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs6.flatten()[cat].text(
            0.6,
            0.6,
            en_bin_str[cat],
            transform=axs6.flatten()[cat].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'extShower_en' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig6)

    # do plots in bins of energy: shower energy
    # boundary every 100 GeV
    fig6, axs6 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs6.flatten()

    for cat in range(n_en_cat):
        for key in showerEn_cat_en_dict[cat]:
            axs6.flatten()[cat].hist(showerEn_cat_en_dict[cat][key],
                                    bins=55,
                                    range=(0, 1100),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower energy')
        axs6.flatten()[cat].set_ylabel('# trk')
        axs6.flatten()[cat].set_yscale('log')
        # add a box containing the energy range
        axs6.flatten()[cat].text(
            0.6,
            0.6,
            en_bin_str[cat],
            transform=axs6.flatten()[cat].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'showerEn_en' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig6)

    # do plots in bins of energy: shower eta
    # boundary every 100 GeV
    fig6, axs6 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs6.flatten()

    for cat in range(n_en_cat):
        for key in showerEta_cat_en_dict[cat]:
            axs6.flatten()[cat].hist(showerEta_cat_en_dict[cat][key],
                                    bins=48,
                                    range=(1.5, 3.1),
                                    density=norm,
                                    histtype='step',
                                    label=key
                                    )
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower eta')
        axs6.flatten()[cat].set_ylabel('# trk')
        axs6.flatten()[cat].set_yscale('log')
        # add a box containing the energy range
        axs6.flatten()[cat].text(
            0.6,
            0.6,
            en_bin_str[cat],
            transform=axs6.flatten()[cat].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.5)
            )

    plotname = 'showerEta_en' + nameAppendix + '.png'
    plt.savefig(os.path.join(out_dir, plotname))  #save plot
    plt.close(fig6)

    # do 2D histo for shower extension vs energy
    # inclusive
    for key in showerEn_dict:
        fig7, axs7 = plt.subplots(1, 1, figsize=(8,6), dpi=80, tight_layout=True)
        axs7.hist2d(showerEn_dict[key], extShower_dict[key], bins=25)
        axs7.set_xlabel('shower energy')
        axs7.set_ylabel('shower extension')
        axs7.set_title(key)
        plt.savefig(os.path.join(out_dir, 'extShower_vs_en_'+key+'.png'))  #save plot
        plt.close(fig7)
