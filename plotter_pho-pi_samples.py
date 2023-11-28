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
    # 5 high density eta bins: 1.65 - 1.75 - 1.85 - 1.95 - 2.05 - 2.15  
    # 3 low density eta bins: 2.15 - 2.35 - 2.55 - 2.75
    # define list of eta bins boundaries
    bin_ed_list = [1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.35, 2.55, 2.75]

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
        #print('ERROR: eta out of range')
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
    extShower_arr_pho = [] # array for shower extension
    showerEn_arr_pho  = [] # array for energy

    # define array of min_clusL and max_clusL for each eta bin
    min_clusL_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    max_clusL_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    extShower_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    showerEn_arr_cat_eta_pho  = [[] for i in range(n_eta_cat)]

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en_pho = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en_pho = [[] for i in range(n_en_cat)]
    extShower_arr_cat_en_pho = [[] for i in range(n_en_cat)]

    # define array of strings for eta bins
    eta_bin_str = ['1.65 < eta < 1.75', '1.75 < eta < 1.85', '1.85 < eta < 1.95', '1.95 < eta < 2.05', '2.05 < eta < 2.15', '2.15 < eta < 2.35', '2.35 < eta < 2.55', '2.55 < eta < 2.75']
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
            extShower_arr_pho.append(abs(trackster_pho[5]-trackster_pho[4]))
            showerEn_arr_pho.append(trackster_pho[2])
            
            # get the eta category number
            cat_eta_n_pho = _divide_eta_categ(trackster_pho[0]) 
            # divide in eta bins
            min_clusL_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[4])
            max_clusL_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[5])
            extShower_arr_cat_eta_pho[cat_eta_n_pho].append(abs(trackster_pho[5]-trackster_pho[4]))
            showerEn_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[2])
            
            # get the energy category number
            cat_en_n_pho = _divide_en_categ(trackster_pho[2])
            # divide in energy bins
            min_clusL_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[4])
            max_clusL_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[5])
            extShower_arr_cat_en_pho[cat_en_n_pho].append(abs(trackster_pho[5]-trackster_pho[4]))


            
    ### PIONS
    min_clusL_arr_pi = [] # array of all min_clusL
    max_clusL_arr_pi = [] # array of all max_clusL
    extShower_arr_pi = [] # array for shower extension
    showerEn_arr_pi  = [] # array for energy

    # define array of min_clusL and max_clusL for each eta bin
    min_clusL_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    max_clusL_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    extShower_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    showerEn_arr_cat_eta_pi  = [[] for i in range(n_eta_cat)]

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en_pi = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en_pi = [[] for i in range(n_en_cat)]
    extShower_arr_cat_en_pi = [[] for i in range(n_en_cat)]


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
            extShower_arr_pi.append(abs(trackster_pi[5]-trackster_pi[4]))
            showerEn_arr_pi.append(trackster_pi[2])
            
            # get the eta category number
            cat_eta_n_pi = _divide_eta_categ(trackster_pi[0]) 
            # divide in eta bins
            min_clusL_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[4])
            max_clusL_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[5])
            extShower_arr_cat_eta_pi[cat_eta_n_pi].append(abs(trackster_pi[5]-trackster_pi[4]))
            showerEn_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[2])
            
            # get the energy category number
            cat_en_n_pi = _divide_en_categ(trackster_pi[2])
            # divide in energy bins
            min_clusL_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[4])
            max_clusL_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[5])
            extShower_arr_cat_en_pi[cat_en_n_pi].append(abs(trackster_pi[5]-trackster_pi[4]))




    # histos min-max L - inclusive
    fig, axs = plt.subplots(1, 2, figsize=(20,10), dpi=80, tight_layout=True)
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



    # hist of shower extension both - inclusive
    fig0, axs0 = plt.subplots(1, 1, figsize=(20,10), dpi=80, tight_layout=True)
    axs0.hist(extShower_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$') 
    axs0.hist(extShower_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs0.legend()
    axs0.set_xlabel('shower extension')
    axs0.set_ylabel('# trk')
    plt.savefig(os.path.join(out_dir, 'extShower.png')) #save plot
    plt.close(fig0)
    
   


    ### do plots in bins of eta: min_clusL
    # boundary between low and high density: 2.15 eta
    fig1, axs1 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs1.flatten()

    for cat in range(n_eta_cat):
        axs1.flatten()[cat].hist(min_clusL_arr_cat_eta_pi[cat], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
        axs1.flatten()[cat].hist(min_clusL_arr_cat_eta_pho[cat], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
        axs1.flatten()[cat].legend()
        axs1.flatten()[cat].set_xlabel('min clusL')
        axs1.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs1.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs1.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.savefig(os.path.join(out_dir, 'minL_eta.png')) #save plot
    plt.close(fig1)


    ### do plots in bins of eta: max_clusL
    # boundary between low and high density: 2.15 eta
    fig2, axs2 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs2.flatten()

    for cat in range(n_eta_cat):
        axs2.flatten()[cat].hist(max_clusL_arr_cat_eta_pi[cat], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
        axs2.flatten()[cat].hist(max_clusL_arr_cat_eta_pho[cat], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
        axs2.flatten()[cat].legend()
        axs2.flatten()[cat].set_xlabel('max clusL')
        axs2.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs2.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs2.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.savefig(os.path.join(out_dir, 'maxL_eta.png')) #save plot
    plt.close(fig2)


    # do plots in bins of eta: shower extension
    # boundary between low and high density: 2.15 eta
    fig3, axs3 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()

    for cat in range(n_eta_cat):
        axs3.flatten()[cat].hist(extShower_arr_cat_eta_pi[cat], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
        axs3.flatten()[cat].hist(extShower_arr_cat_eta_pho[cat], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower extension')
        axs3.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs3.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs3.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'extShower_eta.png')) #save plot
    plt.close(fig3)


    # do plots in bins of energy: min_clusL
    # boundary every 100 GeV
    fig4, axs4 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs4.flatten()

    for cat in range(n_en_cat):
        axs4.flatten()[cat].hist(min_clusL_arr_cat_en_pi[cat], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
        axs4.flatten()[cat].hist(min_clusL_arr_cat_en_pho[cat], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
        axs4.flatten()[cat].legend()
        axs4.flatten()[cat].set_xlabel('min clusL')
        axs4.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs4.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs4.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.savefig(os.path.join(out_dir, 'minL_en.png')) #save plot
    plt.close(fig4)


    # do plots in bins of energy: max_clusL
    # boundary every 100 GeV
    fig5, axs5 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs5.flatten()

    for cat in range(n_en_cat):
        axs5.flatten()[cat].hist(max_clusL_arr_cat_en_pi[cat], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
        axs5.flatten()[cat].hist(max_clusL_arr_cat_en_pho[cat], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
        axs5.flatten()[cat].legend()
        axs5.flatten()[cat].set_xlabel('max clusL')
        axs5.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs5.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs5.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'maxL_en.png')) #save plot
    plt.close(fig4)


    # do plots in bins of energy: shower extension
    # boundary every 100 GeV
    fig6, axs6 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs6.flatten()

    for cat in range(n_en_cat):
        axs6.flatten()[cat].hist(extShower_arr_cat_en_pi[cat], bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
        axs6.flatten()[cat].hist(extShower_arr_cat_en_pho[cat], bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower extension')
        axs6.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs6.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs6.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'extShower_en.png')) #save plot
    plt.close(fig6)


    # do 2D histo for shower extension vs energy
    # inclusive
    fig7, axs7 = plt.subplots(1, 2, figsize=(20,10), dpi=80, tight_layout=True)
    axs7[0].hist2d(showerEn_arr_pi, extShower_arr_pi, bins=30, cmap='Greens')
    axs7[0].set_xlabel('shower energy')
    axs7[0].set_ylabel('shower extension')
    axs7[0].set_title(r'$\pi$')
    axs7[1].hist2d(showerEn_arr_pho, extShower_arr_pho, bins=30, cmap='Oranges')
    axs7[1].set_xlabel('shower energy')
    axs7[1].set_ylabel('shower extension')
    axs7[1].set_title(r'$\gamma$')
    plt.savefig(os.path.join(out_dir, 'extShower_vs_en.png')) #save plot
    plt.close(fig7)

    # do 2D histo for shower extension vs energy
    # in bins of eta
    # photons
    fig8, axs8 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs8.flatten()
    for cat in range(n_eta_cat):
        axs8.flatten()[cat].hist2d(showerEn_arr_cat_eta_pho[cat], extShower_arr_cat_eta_pho[cat], bins=30, cmap='Oranges')
        axs8.flatten()[cat].set_xlabel('shower energy')
        axs8.flatten()[cat].set_ylabel('shower extension')
        #axs8.flatten()[cat].set_title(eta_bin_str[cat])
        # add box for eta range
        axs8.flatten()[cat].text(0.6, 0.2, eta_bin_str[cat], transform=axs8.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.savefig(os.path.join(out_dir, 'extShower_vs_en_eta_pho.png')) #save plot
    plt.close(fig8)
    # pions
    fig9, axs9 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs9.flatten()
    for cat in range(n_eta_cat):
        axs9.flatten()[cat].hist2d(showerEn_arr_cat_eta_pi[cat], extShower_arr_cat_eta_pi[cat], bins=30, cmap='Greens')
        axs9.flatten()[cat].set_xlabel('shower energy')
        axs9.flatten()[cat].set_ylabel('shower extension')
        #axs9.flatten()[cat].set_title(eta_bin_str[cat])
        # add box for eta range
        axs9.flatten()[cat].text(0.6, 0.2, eta_bin_str[cat], transform=axs9.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.savefig(os.path.join(out_dir, 'extShower_vs_en_eta_pi.png')) #save plot
    plt.close(fig9)



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
            en_sum_perL_arr = [np.sum(i, dtype=np.float32) for i in en_arr]

            # compute the energy fraction per layer 
            # divide by the total energy of the trackster
            en_frac_arr = [i/trackster[2] for i in en_sum_perL_arr] # there is an energy fraction per each layer
            
            en_arr_frac_matrix.append(en_frac_arr) # append the array of energy fraction per layer to the list of arrays

    return np.array(en_arr_frac_matrix) # convert the list of arrays in a numpy object (THE ULTIMATE MATRIX!)


def _energy_profile_per_cat(data_list: List[Data], cat_type: str, n_cat: int, n_layers: int = 48) -> np.ndarray:

    # define array of energy fraction per layer per each category
    en_arr_frac_matrix = [[] for _ in range(n_cat)]

    # loop over files in data list
    for i_file in data_list:
        # loop over all events in one file
        for i_evt in i_file:

            # energy array for each Layer of all LC
            # create an array of 48 empty arrays
            en_arr = [[] for _ in range(n_layers)]

            # --- read 2D objects
            layerClusterMatrix = i_evt.clus2d_feat.numpy() # transform the tensor in numpy array

            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features: 
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster = i_evt.clus3d_feat.numpy() # transform the tensor in numpy array

            # get the category number
            if cat_type == 'eta_categ':
                cat_number = _divide_eta_categ(trackster[0])
            elif cat_type == 'en_categ':
                cat_number = _divide_en_categ(trackster[2])
            else:
                raise Exception('category type not recognized')
            

            # loop over LC of the event
            for i_LC in range(len(layerClusterMatrix)): #loop over matrix rows
                en_arr[int(layerClusterMatrix[i_LC,5])].append(layerClusterMatrix[i_LC,3]) # fill array of all energies of all LCs in all events. there is one array per Layer

            # compute the sum of the energy of all LC per Layer
            en_sum_perL_arr = [np.sum(i, dtype=np.float32) for i in en_arr]

            # compute the energy fraction per layer 
            # divide by the total energy of the trackster
            en_frac_arr = [i/trackster[2] for i in en_sum_perL_arr] # there is an energy fraction per each layer
            
            en_arr_frac_matrix[cat_number].append(en_frac_arr) # append the array of energy fraction per layer to the list of arrays per each category

    # pad the arrays such that the final shape is n_cat x n_events x n_layers,
    # where n_events is the maximum number of events in a category
    max_n_events = max([len(i) for i in en_arr_frac_matrix]) # maximum number of events in a category
    en_arr_frac_matrix_np = np.zeros((n_cat, max_n_events, n_layers)) # create a numpy array of zeros with the final shape
    for n, cat_matrix in enumerate(en_arr_frac_matrix):

        if len(cat_matrix) == 0:
            continue

        M = np.pad(cat_matrix, ((0,max_n_events-len(cat_matrix)),(0,0)), 'constant', constant_values=(0,0)) # pad the arrays with zeros to have the max number of events as rows. leave the layer dimension intact
        en_arr_frac_matrix_np[n] = M # fill the numpy array

    return en_arr_frac_matrix_np # (THE ULTIMATE MATRIX per category!)


def doENprofile(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str, n_layers: int = 48, n_eta_cat : int =8, n_en_cat : int =9) -> None:
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
    binEdges_list = np.arange(0, n_layers) # this way I have 49 bins from 0 to 48 : 48 bins = 48 layers
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
   

    # energy profile per CATEGORY
    en_arr_frac_pho_matrix_cat_eta = _energy_profile_per_cat(data_list_pho, 'eta_categ', n_eta_cat)
    en_arr_frac_pho_matrix_cat_en = _energy_profile_per_cat(data_list_pho, 'en_categ', n_en_cat)
    
    en_arr_frac_pi_matrix_cat_eta = _energy_profile_per_cat(data_list_pi, 'eta_categ', n_eta_cat)
    en_arr_frac_pi_matrix_cat_en = _energy_profile_per_cat(data_list_pi, 'en_categ', n_en_cat)

    # compute the mean of the energy fraction over the tracksters
    # sum over the events axis
    # (this object has shape [ncat][nevents][nlayer])
    # the most left index is the most internal (thus =0)
    print('creating fraction matrix for category...')
    en_mean_arr_pho_cat_eta = en_arr_frac_pho_matrix_cat_eta.mean(axis=1)
    en_mean_arr_pi_cat_eta = en_arr_frac_pi_matrix_cat_eta.mean(axis=1)
    en_mean_arr_pho_cat_en = en_arr_frac_pho_matrix_cat_en.mean(axis=1)
    en_mean_arr_pi_cat_en = en_arr_frac_pi_matrix_cat_en.mean(axis=1)

    # plot per category
    # define array of strings for eta bins
    eta_bin_str = ['1.65 < eta < 1.75', '1.75 < eta < 1.85', '1.85 < eta < 1.95', '1.95 < eta < 2.05', '2.05 < eta < 2.15', '2.15 < eta < 2.35', '2.35 < eta < 2.55', '2.55 < eta < 2.75']
    # define array of strings for energy bins
    en_bin_str = ['0 < E < 100 GeV', '100 < E < 200 GeV', '200 < E < 300 GeV', '300 < E < 400 GeV', '400 < E < 500 GeV', '500 < E < 600 GeV', '600 < E < 700 GeV', '700 < E < 800 GeV', 'E > 800 GeV']

    # eta categories
    fig2, axs2 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs2.flatten()
    for cat in range(n_eta_cat):
        axs2.flatten()[cat].plot(binEdges_list, en_mean_arr_pi_cat_eta[cat], linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
        axs2.flatten()[cat].plot(binEdges_list, en_mean_arr_pho_cat_eta[cat], linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
        axs2.flatten()[cat].legend()
        axs2.flatten()[cat].set_xlabel('Layer Number')
        axs2.flatten()[cat].set_ylabel('Energy fraction mean')
        # add a box containing the eta range
        axs2.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs2.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'en_profile_eta.png')) #save plot
    plt.close(fig2)

    # energy categories
    fig3, axs3 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()
    for cat in range(n_en_cat):
        axs3.flatten()[cat].plot(binEdges_list, en_mean_arr_pi_cat_en[cat], linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
        axs3.flatten()[cat].plot(binEdges_list, en_mean_arr_pho_cat_en[cat], linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('Layer Number')
        axs3.flatten()[cat].set_ylabel('Energy fraction mean')
        # add a box containing the energy range
        axs3.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs3.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'en_profile_en.png')) #save plot
    plt.close(fig3)


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

    fig, axs = plt.subplots(1, 3, figsize=(20,12), dpi=80, tight_layout=True)
    # plot eta
    axs[0].hist(gun_matrix_pi[:,0], bins=50, range=(1.2,3.2),
                color='green', alpha=0.4, label=r'$\pi$')
    axs[0].hist(gun_matrix_pho[:,0], bins=50, range=(1.2,3.2),
                color='orange', alpha=0.4, label=r'$\gamma$')
    axs[0].legend()
    axs[0].set_xlabel('eta')
    axs[0].set_ylabel('# trk')
    # plot phi
    axs[1].hist(gun_matrix_pi[:,1], bins=50, range=(-4.,4.), 
                color='green', alpha=0.4, label=r'$\pi$')
    axs[1].hist(gun_matrix_pho[:,1], bins=50, range=(-4.,4.),
                color='orange', alpha=0.4, label=r'$\gamma$')
    axs[1].legend()
    axs[1].set_xlabel('phi')
    axs[1].set_ylabel('# trk')
    # plot energy
    axs[2].hist(gun_matrix_pi[:,2], bins=50, range=(0.,1200.),
                color='green', alpha=0.4, label=r'$\pi$')
    axs[2].hist(gun_matrix_pho[:,2], bins=50, range=(0.,1200.),
                color='orange', alpha=0.4, label=r'$\gamma$')
    axs[2].legend()
    axs[2].set_xlabel('energy')
    axs[2].set_ylabel('# trk')

    plt.savefig(os.path.join(out_dir, 'gunFeats.png')) #save plot
    plt.close(fig)
    

# function to plot LC multiplicity
# we want the average multiplicity of LC per layer
def doMultiplicityPlots(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str, n_layers: int = 48) -> None: 

    # --- pions
    mult_matrix_pi = [[] for _ in range(n_layers)] # array of arrays of multiplicity per layer (per each event)
    mult_arr_pi = [] # array of multiplicity per layer
    
    for i_file in data_list_pi:
        for i_evt in i_file:
            
            # create array for each Layer 
            counter_arr_pi = [0 for _ in range(n_layers)] # array of zeros of length n_layers

            # --- read 2D objects
            # layerClusterMatrix = matrix of all the LayerClusters in the file
            # LayerCluster features: clusX,clusY,clusZ,clusE,clusT,clusL
            # (number of rows) 
            # with their features (number of columns)
            # there is one matrix of this kind for each event of the loadfile_pi
            layerClusterMatrix_pi = i_evt.clus2d_feat.numpy()

            # loop over the LC of the event
            for i_LC in range(len(layerClusterMatrix_pi)):
                counter_arr_pi[int(layerClusterMatrix_pi[i_LC,5])] += 1 # count how many LC per layer; there is one element per layer

            # append the array of multiplicity per layer to the list of arrays
            for i_L in range(n_layers):
                mult_matrix_pi[i_L].append(counter_arr_pi[i_L])

    mult_matrix_pi = np.array(mult_matrix_pi)
    print(mult_matrix_pi.shape)    

    for i_L in range(n_layers):
        mult_arr_pi.append(np.mean(mult_matrix_pi[i_L]))

    print(mult_arr_pi)


    # --- photons
    mult_matrix_pho = [[] for _ in range(n_layers)] # array of arrays of multiplicity per layer (per each event)
    mult_arr_pho = [] # array of multiplicity per layer

    for i_file in data_list_pho:
        for i_evt in i_file:
            
            # create array for each Layer 
            counter_arr_pho = [0 for _ in range(n_layers)] # array of zeros of length n_layers

            # --- read 2D objects
            layerClusterMatrix_pho = i_evt.clus2d_feat.numpy()

            # loop over the LC of the event
            for i_LC in range(len(layerClusterMatrix_pho)):
                counter_arr_pho[int(layerClusterMatrix_pho[i_LC,5])] += 1 # count how many LC per layer; there is one element per layer

            # append the array of multiplicity per layer to the list of arrays
            for i_L in range(n_layers):
                mult_matrix_pho[i_L].append(counter_arr_pho[i_L])

    mult_matrix_pho = np.array(mult_matrix_pho)
    print(mult_matrix_pho.shape)

    for i_L in range(n_layers):
        mult_arr_pho.append(np.mean(mult_matrix_pho[i_L]))

    print(mult_arr_pho)


    # plot multiplicity
    fig, ax = plt.subplots(figsize=(17,10), dpi=80, tight_layout=True)
    layer_list = np.arange(0, n_layers) # this way I have 48 bins from 0 to 48 : 48 bins = 48 layers 
    ax.plot(layer_list, mult_arr_pi, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    ax.plot(layer_list, mult_arr_pho, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    ax.legend()
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Multiplicity mean')

    plt.savefig(os.path.join(out_dir, 'mult.png')) #save plot
    plt.close(fig)


    # plot multiplicity per layer
    for i_L in range(n_layers):
        fig1, ax1 = plt.subplots(figsize=(12,8), dpi=80, tight_layout=True)
        ax1.hist(mult_matrix_pi[i_L], color='green', alpha=0.4, label=r'$\pi$')
        ax1.hist(mult_matrix_pho[i_L], color='orange', alpha=0.4, label=r'$\gamma$')
        ax1.legend()
        ax1.set_xlabel('Multiplicity')
        ax1.set_ylabel('# LC')
        ax1.set_title('Layer '+str(i_L))

        plt.savefig(os.path.join(out_dir, 'mult_layer'+str(i_L)+'.png')) #save plot
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

    doGunPlots(data_list_pho, data_list_pi, out_dir)

    doMultiplicityPlots(data_list_pho, data_list_pi, out_dir)