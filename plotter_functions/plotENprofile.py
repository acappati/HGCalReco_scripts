"""
function to plot energy profile of the trackster
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
from plotter_functions import _energy_profile, _energy_profile_per_cat

plt.style.use(hep.style.CMS)
mpl.use('agg')



def doENprofile(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str, n_layers: int = 48, n_eta_cat : int =8, n_en_cat : int =9) -> None:
    """
    function to compute the energy profile
    """
    ### PHOTONS
    print('[plotENprofile]: creating fraction matrix for photons...')
    en_arr_frac_pho_matrix = _energy_profile(data_list_pho)

    ### PIONS
    print('[plotENprofile]: creating fraction matrix for pion...')
    en_arr_frac_pi_matrix = _energy_profile(data_list_pi)


    # compute the mean of the energy fraction over the tracksters
    en_mean_arr_pho = en_arr_frac_pho_matrix.mean(axis=0)
    en_mean_arr_pi = en_arr_frac_pi_matrix.mean(axis=0)

    # plot energy profile
    fig, ax = plt.subplots(figsize=(17,10), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, n_layers, 1) #list of 48 elements, from 0 to 47
    ax.plot(binEdges_list, en_mean_arr_pi, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    ax.plot(binEdges_list, en_mean_arr_pho, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    ax.legend()
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Energy fraction mean')

    plt.savefig(os.path.join(out_dir, 'en_profile.png')) #save plot
    plt.close(fig)


    # not super correct
    # # compute the energy fraction in the hadronic part of HGCal
    # # consider the last 22 layers
    # # compute the energy fraction per each trackster
    # fracH_arr_pho = en_arr_frac_pho_matrix[:,26:].sum(axis=1)/en_arr_frac_pho_matrix.sum(axis=1)
    # fracH_arr_pi = en_arr_frac_pi_matrix[:,26:].sum(axis=1)/en_arr_frac_pi_matrix.sum(axis=1)

    # # plot energy fraction in the hadronic part
    # fig1, ax1 = plt.subplots(figsize=(12,8), dpi=80, tight_layout=True)
    # ax1.hist(fracH_arr_pi, bins=50, range=(0.,0.21), density=True, color='green', alpha=0.4, label=r'$\pi$')
    # ax1.hist(fracH_arr_pho, bins=50, range=(0.,0.21), density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    # ax1.legend()
    # ax1.set_yscale('log')
    # ax1.set_xlabel('Energy fraction in CEH')
    # ax1.set_ylabel('# tracksters')

    # plt.savefig(os.path.join(out_dir, 'en_fracH.png')) #save plot
    # plt.close(fig1)


    # energy profile per CATEGORY
    en_arr_frac_pho_matrix_cat_eta = _energy_profile_per_cat(data_list_pho, 'eta_categ', n_eta_cat)
    en_arr_frac_pho_matrix_cat_en = _energy_profile_per_cat(data_list_pho, 'en_categ', n_en_cat)

    en_arr_frac_pi_matrix_cat_eta = _energy_profile_per_cat(data_list_pi, 'eta_categ', n_eta_cat)
    en_arr_frac_pi_matrix_cat_en = _energy_profile_per_cat(data_list_pi, 'en_categ', n_en_cat)

    # compute the mean of the energy fraction over the tracksters
    # sum over the events axis
    # (this object has shape [ncat][nevents][nlayer])
    # the most left index is the most internal (thus =0)
    print('[plotENprofile]: creating fraction matrix for category...')
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
