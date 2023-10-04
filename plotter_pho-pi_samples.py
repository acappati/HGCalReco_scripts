#!/usr/bin/env/python3

## ---
#  script to plot variables from training samples
#  run with: python3 plotter_pho-pi_samples.py
## ---

import glob
import math
import optparse
import os
import os.path as osp
import sys
from datetime import date
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import torch_geometric
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

plt.style.use(hep.style.CMS)
mpl.use('agg')




def openFiles(path: str, desc: str = 'Opening files') -> List[Data]:
    """
    Function to open files

    Arguments
    ---------
    path : str
        Path to the input files
    desc : str
        Description for the tqdm progress bar

    Returns
    -------
    data_list : List[Data]
        List of Data objects
    """
    data_list = []
    filename_list = [f for f in glob.glob(path + 'data_*.pt')]
    filename_list = tqdm(filename_list, desc=desc, unit='file(s)')
    for i in filename_list:
        idx = torch.load(i)
        data_list.append(idx)
    return data_list


def _divide_eta_categ(trackster_eta, n_eta_cat : int =8) -> int:
    """
    function to divide in eta bins
    """
    # define list of eta bins numbers from 0 to n_eta_cat-1
    bin_n = [i for i in range(n_eta_cat)]
    
    # boundary between low and high density: 2.15 eta
    # 5 high density eta bins: 1.5 - 1.63 - 1.76 - 1.89 - 2.02 - 2.15 
    # 3 low density eta bins: 2.15 - 2.48 - 2.81 - 3.14
    # define list of eta bins boundaries
    bin_ed_list = [1.5, 1.63, 1.76, 1.89, 2.02, 2.15, 2.48, 2.81, 3.14]

    if trackster_eta >= bin_ed_list[0] and trackster_eta < bin_ed_list[1]:
        return bin_n[0]
    elif trackster_eta >= bin_ed_list[1] and trackster_eta < bin_ed_list[2]:
        return bin_n[1]
    elif trackster_eta >= bin_ed_list[2] and trackster_eta < bin_ed_list[3]:
        return bin_n[2]
    elif trackster_eta >= bin_ed_list[3] and trackster_eta < bin_ed_list[4]:
        return bin_n[3]
    elif trackster_eta >= bin_ed_list[4] and trackster_eta <= bin_ed_list[5]:
        return bin_n[4]
    elif trackster_eta > bin_ed_list[5] and trackster_eta < bin_ed_list[6]:
        return bin_n[5]
    elif trackster_eta >= bin_ed_list[6] and trackster_eta < bin_ed_list[7]:
        return bin_n[6]
    elif trackster_eta >= bin_ed_list[7] and trackster_eta <= bin_ed_list[8]:
        return bin_n[7]
    else:
        print('ERROR: eta out of range')
        return -1
    

def _divide_en_categ(trackster_en, n_en_cat : int =9) -> int:
    """
    function to divide in energy bins
    """
    # define list of energy bins numbers from 0 to n_en_cat-1
    bin_n = [i for i in range(n_en_cat)]

    # boundary every 100 GeV
    # energy bins: 0 - 100 - 200 - 300 - 400 - 500 - 600 - 700 - 800 - inf
    bin_ed_list = [0, 100, 200, 300, 400, 500, 600, 700, 800]

    if trackster_en >= bin_ed_list[0] and trackster_en < bin_ed_list[1]:
        return bin_n[0]
    elif trackster_en >= bin_ed_list[1] and trackster_en < bin_ed_list[2]:
        return bin_n[1]
    elif trackster_en >= bin_ed_list[2] and trackster_en < bin_ed_list[3]:
        return bin_n[2]
    elif trackster_en >= bin_ed_list[3] and trackster_en < bin_ed_list[4]:
        return bin_n[3]
    elif trackster_en >= bin_ed_list[4] and trackster_en < bin_ed_list[5]:
        return bin_n[4]
    elif trackster_en >= bin_ed_list[5] and trackster_en < bin_ed_list[6]:
        return bin_n[5]
    elif trackster_en >= bin_ed_list[6] and trackster_en < bin_ed_list[7]:
        return bin_n[6]
    elif trackster_en >= bin_ed_list[7] and trackster_en < bin_ed_list[8]:
        return bin_n[7]
    elif trackster_en >= bin_ed_list[8]:
        return bin_n[8]
    else:
        print('ERROR: energy out of range')
        return -1
       


def doHisto(data_list_pho, data_list_pi, out_dir, n_eta_cat : int =8, n_en_cat : int =9):
    """
    function to do histograms from the training samples
    """

    ### PHOTONS
    min_clusL_arr_pho = [] # array of all min_clusL
    max_clusL_arr_pho = [] # array of all max_clusL

    # define array of min_clusL and max_clusL for each eta bin
    min_clusL_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    max_clusL_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en_pho = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en_pho = [[] for i in range(n_en_cat)]

    # define array of strings for eta bins
    eta_bin_str = ['1.5 < eta < 1.63', '1.63 < eta < 1.76', '1.76 < eta < 1.89', '1.89 < eta < 2.02', '2.02 < eta < 2.15', '2.15 < eta < 2.48', '2.48 < eta < 2.81', '2.81 < eta < 3.14']
    # define array of strings for energy bins
    en_bin_str = ['0 < E < 100 GeV', '100 < E < 200 GeV', '200 < E < 300 GeV', '300 < E < 400 GeV', '400 < E < 500 GeV', '500 < E < 600 GeV', '600 < E < 700 GeV', '700 < E < 800 GeV', 'E > 800 GeV']


    # loop over files in data list
    for i_file_pho in data_list_pho:
        # loop over all events in one file
        for i_evt_pho in i_file_pho:

            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features: 
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster_pho = i_evt_pho.clus3d_feat.numpy() # transform the tensor in numpy array

            # fill array of all min and max cluster Layer
            min_clusL_arr_pho.append(trackster_pho[4]) 
            max_clusL_arr_pho.append(trackster_pho[5])
            
            # get the eta category number
            cat_eta_n_pho = _divide_eta_categ(trackster_pho[0]) 
            # divide in eta bins
            min_clusL_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[4])
            max_clusL_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[5])
            
            # get the energy category number
            cat_en_n_pho = _divide_en_categ(trackster_pho[2])
            # divide in energy bins
            min_clusL_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[4])
            max_clusL_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[5])

            
    
    ### PIONS
    min_clusL_arr_pi = [] # array of all min_clusL
    max_clusL_arr_pi = [] # array of all max_clusL

    # define array of min_clusL and max_clusL for each eta bin
    min_clusL_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    max_clusL_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en_pi = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en_pi = [[] for i in range(n_en_cat)]


    # loop over files in data list
    for i_file_pi in data_list_pi:
        # loop over all events in one file
        for i_evt_pi in i_file_pi:

            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features: 
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster_pi = i_evt_pi.clus3d_feat.numpy() # transform the tensor in numpy array

             # fill array of all min and max cluster Layer
            min_clusL_arr_pi.append(trackster_pi[4]) 
            max_clusL_arr_pi.append(trackster_pi[5])
            
            # get the eta category number
            cat_eta_n_pi = _divide_eta_categ(trackster_pi[0]) 
            # divide in eta bins
            min_clusL_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[4])
            max_clusL_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[5])
            
            # get the energy category number
            cat_en_n_pi = _divide_en_categ(trackster_pi[2])
            # divide in energy bins
            min_clusL_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[4])
            max_clusL_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[5])




    # histos min-max L - inclusive
    fig, axs = plt.subplots(1, 2, figsize=(20,12), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, 47) # this way I have 48 bins from 0 to 47 : 48 bins = 48 layers 

    # hist of min_clusL both
    axs[0].hist(min_clusL_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs[0].hist(min_clusL_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[0].legend()        
    axs[0].set_xlabel('min clusL')
    axs[0].set_ylabel('# trk')

    # hist of max_clusL both
    axs[1].hist(max_clusL_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs[1].hist(max_clusL_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[1].legend()        
    axs[1].set_xlabel('max clusL')
    axs[1].set_ylabel('# trk')
    
    plt.savefig(os.path.join(out_dir, 'minmaxL.png')) #save plot
    plt.close(fig)


    ### do plots in bins of eta: min_clusL
    # boundary between low and high density: 2.15 eta
    fig1, axs1 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    
    axs1[0][0].hist(min_clusL_arr_cat_eta_pho[0], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[0][0].hist(min_clusL_arr_cat_eta_pi[0], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[0][0].legend()
    axs1[0][0].set_xlabel('min clusL')
    axs1[0][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[0][0].text(0.7, 0.6, eta_bin_str[0], transform=axs1[0][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[0][1].hist(min_clusL_arr_cat_eta_pho[1], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[0][1].hist(min_clusL_arr_cat_eta_pi[1], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[0][1].legend()
    axs1[0][1].set_xlabel('min clusL')
    axs1[0][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[0][1].text(0.7, 0.6, eta_bin_str[1], transform=axs1[0][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[1][0].hist(min_clusL_arr_cat_eta_pho[2], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[1][0].hist(min_clusL_arr_cat_eta_pi[2], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[1][0].legend()
    axs1[1][0].set_xlabel('min clusL')
    axs1[1][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[1][0].text(0.7, 0.6, eta_bin_str[2], transform=axs1[1][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[1][1].hist(min_clusL_arr_cat_eta_pho[3], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[1][1].hist(min_clusL_arr_cat_eta_pi[3], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[1][1].legend()
    axs1[1][1].set_xlabel('min clusL')
    axs1[1][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[1][1].text(0.7, 0.6, eta_bin_str[3], transform=axs1[1][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[2][0].hist(min_clusL_arr_cat_eta_pho[4], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[2][0].hist(min_clusL_arr_cat_eta_pi[4], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[2][0].legend()
    axs1[2][0].set_xlabel('min clusL')
    axs1[2][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[2][0].text(0.7, 0.6, eta_bin_str[4], transform=axs1[2][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[2][1].hist(min_clusL_arr_cat_eta_pho[5], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[2][1].hist(min_clusL_arr_cat_eta_pi[5], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[2][1].legend()
    axs1[2][1].set_xlabel('min clusL')
    axs1[2][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[2][1].text(0.7, 0.6, eta_bin_str[5], transform=axs1[2][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[3][0].hist(min_clusL_arr_cat_eta_pho[6], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[3][0].hist(min_clusL_arr_cat_eta_pi[6], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[3][0].legend()
    axs1[3][0].set_xlabel('min clusL')
    axs1[3][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[3][0].text(0.7, 0.6, eta_bin_str[6], transform=axs1[3][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs1[3][1].hist(min_clusL_arr_cat_eta_pho[7], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs1[3][1].hist(min_clusL_arr_cat_eta_pi[7], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs1[3][1].legend()
    axs1[3][1].set_xlabel('min clusL')
    axs1[3][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs1[3][1].text(0.7, 0.6, eta_bin_str[7], transform=axs1[3][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'minL_eta.png')) #save plot
    plt.close(fig1)


    ### do plots in bins of eta: max_clusL
    # boundary between low and high density: 2.15 eta
    fig2, axs2 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    
    axs2[0][0].hist(max_clusL_arr_cat_eta_pho[0], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[0][0].hist(max_clusL_arr_cat_eta_pi[0], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[0][0].legend()
    axs2[0][0].set_xlabel('max clusL')
    axs2[0][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[0][0].text(0.7, 0.6, eta_bin_str[0], transform=axs2[0][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[0][1].hist(max_clusL_arr_cat_eta_pho[1], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[0][1].hist(max_clusL_arr_cat_eta_pi[1], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[0][1].legend()
    axs2[0][1].set_xlabel('max clusL')
    axs2[0][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[0][1].text(0.7, 0.6, eta_bin_str[1], transform=axs2[0][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[1][0].hist(max_clusL_arr_cat_eta_pho[2], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[1][0].hist(max_clusL_arr_cat_eta_pi[2], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[1][0].legend()
    axs2[1][0].set_xlabel('max clusL')
    axs2[1][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[1][0].text(0.7, 0.6, eta_bin_str[2], transform=axs2[1][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[1][1].hist(max_clusL_arr_cat_eta_pho[3], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[1][1].hist(max_clusL_arr_cat_eta_pi[3], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[1][1].legend()
    axs2[1][1].set_xlabel('max clusL')
    axs2[1][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[1][1].text(0.7, 0.6, eta_bin_str[3], transform=axs2[1][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[2][0].hist(max_clusL_arr_cat_eta_pho[4], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[2][0].hist(max_clusL_arr_cat_eta_pi[4], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[2][0].legend()
    axs2[2][0].set_xlabel('max clusL')
    axs2[2][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[2][0].text(0.7, 0.6, eta_bin_str[4], transform=axs2[2][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[2][1].hist(max_clusL_arr_cat_eta_pho[5], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[2][1].hist(max_clusL_arr_cat_eta_pi[5], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[2][1].legend()
    axs2[2][1].set_xlabel('max clusL')
    axs2[2][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[2][1].text(0.7, 0.6, eta_bin_str[5], transform=axs2[2][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[3][0].hist(max_clusL_arr_cat_eta_pho[6], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[3][0].hist(max_clusL_arr_cat_eta_pi[6], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[3][0].legend()
    axs2[3][0].set_xlabel('max clusL')
    axs2[3][0].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[3][0].text(0.7, 0.6, eta_bin_str[6], transform=axs2[3][0].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs2[3][1].hist(max_clusL_arr_cat_eta_pho[7], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs2[3][1].hist(max_clusL_arr_cat_eta_pi[7], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs2[3][1].legend()
    axs2[3][1].set_xlabel('max clusL')
    axs2[3][1].set_ylabel('# trk')
    # add a box containing the eta range
    axs2[3][1].text(0.7, 0.6, eta_bin_str[7], transform=axs2[3][1].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'maxL_eta.png')) #save plot
    plt.close(fig2)


    # do plots in bins of energy: min_clusL
    # boundary every 100 GeV
    fig3, axs3 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)

    axs3[0][0].hist(min_clusL_arr_cat_en_pho[0], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[0][0].hist(min_clusL_arr_cat_en_pi[0], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[0][0].legend()
    axs3[0][0].set_xlabel('min clusL')
    axs3[0][0].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[0][0].text(0.6, 0.6, en_bin_str[0], transform=axs3[0][0].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[0][1].hist(min_clusL_arr_cat_en_pho[1], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[0][1].hist(min_clusL_arr_cat_en_pi[1], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[0][1].legend()
    axs3[0][1].set_xlabel('min clusL')
    axs3[0][1].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[0][1].text(0.6, 0.6, en_bin_str[1], transform=axs3[0][1].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[0][2].hist(min_clusL_arr_cat_en_pho[2], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[0][2].hist(min_clusL_arr_cat_en_pi[2], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[0][2].legend()
    axs3[0][2].set_xlabel('min clusL')
    axs3[0][2].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[0][2].text(0.6, 0.6, en_bin_str[2], transform=axs3[0][2].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[1][0].hist(min_clusL_arr_cat_en_pho[3], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[1][0].hist(min_clusL_arr_cat_en_pi[3], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[1][0].legend()
    axs3[1][0].set_xlabel('min clusL')
    axs3[1][0].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[1][0].text(0.6, 0.6, en_bin_str[3], transform=axs3[1][0].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[1][1].hist(min_clusL_arr_cat_en_pho[4], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[1][1].hist(min_clusL_arr_cat_en_pi[4], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[1][1].legend()
    axs3[1][1].set_xlabel('min clusL')
    axs3[1][1].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[1][1].text(0.6, 0.6, en_bin_str[4], transform=axs3[1][1].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[1][2].hist(min_clusL_arr_cat_en_pho[5], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[1][2].hist(min_clusL_arr_cat_en_pi[5], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[1][2].legend()
    axs3[1][2].set_xlabel('min clusL')
    axs3[1][2].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[1][2].text(0.6, 0.6, en_bin_str[5], transform=axs3[1][2].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[2][0].hist(min_clusL_arr_cat_en_pho[6], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[2][0].hist(min_clusL_arr_cat_en_pi[6], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[2][0].legend()
    axs3[2][0].set_xlabel('min clusL')
    axs3[2][0].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[2][0].text(0.6, 0.6, en_bin_str[6], transform=axs3[2][0].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[2][1].hist(min_clusL_arr_cat_en_pho[7], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[2][1].hist(min_clusL_arr_cat_en_pi[7], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[2][1].legend()
    axs3[2][1].set_xlabel('min clusL')
    axs3[2][1].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[2][1].text(0.6, 0.6, en_bin_str[7], transform=axs3[2][1].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs3[2][2].hist(min_clusL_arr_cat_en_pho[8], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs3[2][2].hist(min_clusL_arr_cat_en_pi[8], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs3[2][2].legend()
    axs3[2][2].set_xlabel('min clusL')
    axs3[2][2].set_ylabel('# trk')
    # add a box containing the energy range
    axs3[2][2].text(0.6, 0.6, en_bin_str[8], transform=axs3[2][2].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'minL_en.png')) #save plot
    plt.close(fig3)


    # do plots in bins of energy: max_clusL
    # boundary every 100 GeV
    fig4, axs4 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)

    axs4[0][0].hist(max_clusL_arr_cat_en_pho[0], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[0][0].hist(max_clusL_arr_cat_en_pi[0], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[0][0].legend()
    axs4[0][0].set_xlabel('max clusL')
    axs4[0][0].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[0][0].text(0.6, 0.6, en_bin_str[0], transform=axs4[0][0].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[0][1].hist(max_clusL_arr_cat_en_pho[1], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[0][1].hist(max_clusL_arr_cat_en_pi[1], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[0][1].legend()
    axs4[0][1].set_xlabel('max clusL')
    axs4[0][1].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[0][1].text(0.6, 0.6, en_bin_str[1], transform=axs4[0][1].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[0][2].hist(max_clusL_arr_cat_en_pho[2], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[0][2].hist(max_clusL_arr_cat_en_pi[2], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[0][2].legend()
    axs4[0][2].set_xlabel('max clusL')
    axs4[0][2].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[0][2].text(0.6, 0.6, en_bin_str[2], transform=axs4[0][2].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[1][0].hist(max_clusL_arr_cat_en_pho[3], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[1][0].hist(max_clusL_arr_cat_en_pi[3], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[1][0].legend()
    axs4[1][0].set_xlabel('max clusL')
    axs4[1][0].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[1][0].text(0.6, 0.6, en_bin_str[3], transform=axs4[1][0].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[1][1].hist(max_clusL_arr_cat_en_pho[4], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[1][1].hist(max_clusL_arr_cat_en_pi[4], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[1][1].legend()
    axs4[1][1].set_xlabel('max clusL')
    axs4[1][1].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[1][1].text(0.6, 0.6, en_bin_str[4], transform=axs4[1][1].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[1][2].hist(max_clusL_arr_cat_en_pho[5], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[1][2].hist(max_clusL_arr_cat_en_pi[5], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[1][2].legend()
    axs4[1][2].set_xlabel('max clusL')
    axs4[1][2].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[1][2].text(0.6, 0.6, en_bin_str[5], transform=axs4[1][2].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[2][0].hist(max_clusL_arr_cat_en_pho[6], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[2][0].hist(max_clusL_arr_cat_en_pi[6], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[2][0].legend()
    axs4[2][0].set_xlabel('max clusL')
    axs4[2][0].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[2][0].text(0.6, 0.6, en_bin_str[6], transform=axs4[2][0].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[2][1].hist(max_clusL_arr_cat_en_pho[7], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[2][1].hist(max_clusL_arr_cat_en_pi[7], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[2][1].legend()
    axs4[2][1].set_xlabel('max clusL')
    axs4[2][1].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[2][1].text(0.6, 0.6, en_bin_str[7], transform=axs4[2][1].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    axs4[2][2].hist(max_clusL_arr_cat_en_pho[8], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs4[2][2].hist(max_clusL_arr_cat_en_pi[8], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs4[2][2].legend()
    axs4[2][2].set_xlabel('max clusL')
    axs4[2][2].set_ylabel('# trk')
    # add a box containing the energy range
    axs4[2][2].text(0.6, 0.6, en_bin_str[8], transform=axs4[2][2].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'maxL_en.png')) #save plot
    plt.close(fig4)


def _energy_profile(data_list: List[Data], n_layers: int = 48) -> np.ndarray:

    en_arr_frac_matrix = [] # array of arrays of energy fraction per layer

    # loop over files in data list
    for i_file in data_list:
        # loop over all events in one file
        for i_evt in i_file:

            # energy array for each Layer of all LC
            # create an array of 48 empty arrays
            en_arr = [[] for _ in range(n_layers)]

            # --- read 2D objects
            # layerClusterMatrix = matrix of all the LayerClusters in the file
            # LayerCluster features: clusX,clusY,clusZ,clusE,clusT,clusL
            # (number of rows) 
            # with their features (number of columns)
            # there is one matrix of this kind for each event of the loadfile_pi
            layerClusterMatrix = i_evt.clus2d_feat.numpy() # transform the tensor in numpy array

            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features: 
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster = i_evt.clus3d_feat.numpy() # transform the tensor in numpy array

            # loop over LC of the event
            for i_LC in range(len(layerClusterMatrix)): #loop over matrix rows
                en_arr[int(layerClusterMatrix[i_LC,5])].append(layerClusterMatrix[i_LC,3]) # fill array of all energies of all LCs in all events. there is one array per Layer

            # compute the sum of the energy of all LC per Layer
            en_sum_perL_arr = [sum(i) for i in en_arr]

            # compute the energy fraction per layer 
            # divide by the total energy of the trackster
            en_frac_arr = [i/trackster[2] for i in en_sum_perL_arr] # there is an energy fraction per each layer
            
            en_arr_frac_matrix.append(en_frac_arr) # append the array of energy fraction per layer to the list of arrays

    return np.array(en_arr_frac_matrix) # convert the list of arrays in a numpy object (THE ULTIMATE MATRIX!)


def doENprofile(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str, n_layers: int = 48) -> None:
    """
    function to compute the energy profile
    """
    ### PHOTONS
    print('creating fraction matrix for photons...')
    en_arr_frac_pho_matrix = _energy_profile(data_list_pho)

    ### PIONS
    print('creating fraction matrix for pion...')
    en_arr_frac_pi_matrix = _energy_profile(data_list_pi)


    # compute the mean of the energy fraction over the tracksters
    en_mean_arr_pho = en_arr_frac_pho_matrix.mean(axis=0)
    en_mean_arr_pi = en_arr_frac_pi_matrix.mean(axis=0)

    # plot energy profile
    fig, ax = plt.subplots(figsize=(17,10), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, 48) # this way I have 49 bins from 0 to 48 : 48 bins = 48 layers
    ax.plot(binEdges_list, en_mean_arr_pi, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    ax.plot(binEdges_list, en_mean_arr_pho, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    ax.legend()
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Energy fraction mean')

    plt.savefig(os.path.join(out_dir, 'en_profile.png')) #save plot
    plt.close(fig)


    # compute the energy fraction in the hadronic part of HGCal
    # consider the last 22 layers
    # compute the energy fraction per each trackster
    fracH_arr_pho = en_arr_frac_pho_matrix[:,27:].sum(axis=1)/en_arr_frac_pho_matrix.sum(axis=1)
    fracH_arr_pi = en_arr_frac_pi_matrix[:,27:].sum(axis=1)/en_arr_frac_pi_matrix.sum(axis=1)

    # plot energy fraction in the hadronic part
    fig1, ax1 = plt.subplots(figsize=(12,8), dpi=80, tight_layout=True)
    ax1.hist(fracH_arr_pi, bins=50, color='green', alpha=0.4, label=r'$\pi$')
    ax1.hist(fracH_arr_pho, bins=50, color='orange', alpha=0.4, label=r'$\gamma$')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy fraction in CEH')
    ax1.set_ylabel('# tracksters')

    plt.savefig(os.path.join(out_dir, 'en_fracH.png')) #save plot
    plt.close(fig1)
   




if __name__ == "__main__" :

    ## output directory
    today = date.today()
    print('Creating output dir...')
    out_dir = str(today)+'_plots'
    os.makedirs(out_dir, exist_ok=True) #check if output dir exist

    ## input files photons
    inpath_pho = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_DATA/ntup_pho_frac0p8/'
    data_list_pho = openFiles(inpath_pho, desc='Loading photon files')

    ## input files pions
    inpath_pi = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_DATA/ntup_pi_frac0p8/'
    data_list_pi = openFiles(inpath_pi, desc='Loading pions files')

    ## plots
    print('doing plots...')
    doHisto(data_list_pho, data_list_pi, out_dir)

    doENprofile(data_list_pho, data_list_pi, out_dir)
