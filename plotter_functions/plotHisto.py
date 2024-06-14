"""
functions to do histograms from the training samples
"""


import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from tqdm import tqdm as tqdm

# import modules
from plotter_functions import _divide_en_categ, _divide_eta_categ

plt.style.use(hep.style.CMS)
mpl.use('agg')




def doHisto(data_list_pho, data_list_pi, out_dir, norm=True, n_eta_cat : int =8, n_en_cat : int =9):
    """
    function to do histograms from the training samples
    """

    ### PHOTONS
    min_clusL_arr_pho = [] # array of all min_clusL
    max_clusL_arr_pho = [] # array of all max_clusL
    extShower_arr_pho = [] # array for shower extension
    showerEn_arr_pho  = [] # array for energy
    showerEta_arr_pho = [] # array for eta

    # define array of min_clusL and max_clusL for each eta bin
    min_clusL_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    max_clusL_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    extShower_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]
    showerEn_arr_cat_eta_pho  = [[] for i in range(n_eta_cat)]
    showerEta_arr_cat_eta_pho = [[] for i in range(n_eta_cat)]

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en_pho = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en_pho = [[] for i in range(n_en_cat)]
    extShower_arr_cat_en_pho = [[] for i in range(n_en_cat)]
    showerEn_arr_cat_en_pho  = [[] for i in range(n_en_cat)]
    showerEta_arr_cat_en_pho = [[] for i in range(n_en_cat)]

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
            showerEta_arr_pho.append(trackster_pho[0])

            # get the eta category number
            cat_eta_n_pho = _divide_eta_categ(trackster_pho[0])
            # divide in eta bins
            min_clusL_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[4])
            max_clusL_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[5])
            extShower_arr_cat_eta_pho[cat_eta_n_pho].append(abs(trackster_pho[5]-trackster_pho[4]))
            showerEn_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[2])
            showerEta_arr_cat_eta_pho[cat_eta_n_pho].append(trackster_pho[0])

            # get the energy category number
            cat_en_n_pho = _divide_en_categ(trackster_pho[2])
            # divide in energy bins
            min_clusL_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[4])
            max_clusL_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[5])
            extShower_arr_cat_en_pho[cat_en_n_pho].append(abs(trackster_pho[5]-trackster_pho[4]))
            showerEn_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[2])
            showerEta_arr_cat_en_pho[cat_en_n_pho].append(trackster_pho[0])



    ### PIONS
    min_clusL_arr_pi = [] # array of all min_clusL
    max_clusL_arr_pi = [] # array of all max_clusL
    extShower_arr_pi = [] # array for shower extension
    showerEn_arr_pi  = [] # array for energy
    showerEta_arr_pi = [] # array for eta

    # define array of min_clusL and max_clusL for each eta bin
    min_clusL_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    max_clusL_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    extShower_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]
    showerEn_arr_cat_eta_pi  = [[] for i in range(n_eta_cat)]
    showerEta_arr_cat_eta_pi = [[] for i in range(n_eta_cat)]

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en_pi = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en_pi = [[] for i in range(n_en_cat)]
    extShower_arr_cat_en_pi = [[] for i in range(n_en_cat)]
    showerEn_arr_cat_en_pi  = [[] for i in range(n_en_cat)]
    showerEta_arr_cat_en_pi = [[] for i in range(n_en_cat)]


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
            showerEta_arr_pi.append(trackster_pi[0])

            # get the eta category number
            cat_eta_n_pi = _divide_eta_categ(trackster_pi[0])
            # divide in eta bins
            min_clusL_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[4])
            max_clusL_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[5])
            extShower_arr_cat_eta_pi[cat_eta_n_pi].append(abs(trackster_pi[5]-trackster_pi[4]))
            showerEn_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[2])
            showerEta_arr_cat_eta_pi[cat_eta_n_pi].append(trackster_pi[0])

            # get the energy category number
            cat_en_n_pi = _divide_en_categ(trackster_pi[2])
            # divide in energy bins
            min_clusL_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[4])
            max_clusL_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[5])
            extShower_arr_cat_en_pi[cat_en_n_pi].append(abs(trackster_pi[5]-trackster_pi[4]))
            showerEn_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[2])
            showerEta_arr_cat_en_pi[cat_en_n_pi].append(trackster_pi[0])




    # histos min-max L - inclusive
    fig, axs = plt.subplots(1, 2, figsize=(20,10), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, 48, 1) # this way I have 48 bins from 0 to 47 : 48 bins = 48 layers

    # hist of min_clusL both
    axs[0].hist(min_clusL_arr_pi, bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
    axs[0].hist(min_clusL_arr_pho, bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[0].legend()
    axs[0].set_xlabel('min clusL')
    axs[0].set_ylabel('# trk')

    # hist of max_clusL both
    axs[1].hist(max_clusL_arr_pi, bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
    axs[1].hist(max_clusL_arr_pho, bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[1].legend()
    axs[1].set_xlabel('max clusL')
    axs[1].set_ylabel('# trk')

    plt.savefig(os.path.join(out_dir, 'minmaxL.png')) #save plot
    plt.close(fig)



    # hist of shower extension both - inclusive
    fig0, axs0 = plt.subplots(1, 1, figsize=(20,10), dpi=80, tight_layout=True)
    axs0.hist(extShower_arr_pi, bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
    axs0.hist(extShower_arr_pho, bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
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
        axs1.flatten()[cat].hist(min_clusL_arr_cat_eta_pi[cat], bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs1.flatten()[cat].hist(min_clusL_arr_cat_eta_pho[cat], bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
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
        axs2.flatten()[cat].hist(max_clusL_arr_cat_eta_pi[cat], bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs2.flatten()[cat].hist(max_clusL_arr_cat_eta_pho[cat], bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
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
        axs3.flatten()[cat].hist(extShower_arr_cat_eta_pi[cat], bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs3.flatten()[cat].hist(extShower_arr_cat_eta_pho[cat], bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower extension')
        axs3.flatten()[cat].set_ylabel('# trk')
        # add a box containing the eta range
        axs3.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs3.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'extShower_eta.png')) #save plot
    plt.close(fig3)


    # do plots in bins of eta: shower energy
    # boundary between low and high density: 2.15 eta
    fig3, axs3 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()

    for cat in range(n_eta_cat):
        axs3.flatten()[cat].hist(showerEn_arr_cat_eta_pi[cat], bins=50, range=(0,1100), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs3.flatten()[cat].hist(showerEn_arr_cat_eta_pho[cat], bins=50, range=(0,1100), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower energy')
        axs3.flatten()[cat].set_ylabel('# trk')
        axs3.flatten()[cat].set_yscale('log')
        # add a box containing the eta range
        axs3.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs3.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'showerEn_eta.png')) #save plot
    plt.close(fig3)

    # do plots in bins of eta: shower eta
    # boundary between low and high density: 2.15 eta
    fig3, axs3 = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs3.flatten()

    for cat in range(n_eta_cat):
        axs3.flatten()[cat].hist(showerEta_arr_cat_eta_pi[cat], bins=20, range=(1.5,3.1), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs3.flatten()[cat].hist(showerEta_arr_cat_eta_pho[cat], bins=20, range=(1.5,3.1), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
        axs3.flatten()[cat].legend()
        axs3.flatten()[cat].set_xlabel('shower eta')
        axs3.flatten()[cat].set_ylabel('# trk')
        axs3.flatten()[cat].set_yscale('log')
        # add a box containing the eta range
        axs3.flatten()[cat].text(0.7, 0.6, eta_bin_str[cat], transform=axs3.flatten()[cat].transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'showerEta_eta.png')) #save plot
    plt.close(fig3)



    # do plots in bins of energy: min_clusL
    # boundary every 100 GeV
    fig4, axs4 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs4.flatten()

    for cat in range(n_en_cat):
        axs4.flatten()[cat].hist(min_clusL_arr_cat_en_pi[cat], bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs4.flatten()[cat].hist(min_clusL_arr_cat_en_pho[cat], bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
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
        axs5.flatten()[cat].hist(max_clusL_arr_cat_en_pi[cat], bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs5.flatten()[cat].hist(max_clusL_arr_cat_en_pho[cat], bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
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
        axs6.flatten()[cat].hist(extShower_arr_cat_en_pi[cat], bins=binEdges_list, range=(0,48), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs6.flatten()[cat].hist(extShower_arr_cat_en_pho[cat], bins=binEdges_list, range=(0,48), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower extension')
        axs6.flatten()[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs6.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs6.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'extShower_en.png')) #save plot
    plt.close(fig6)


    # do plots in bins of energy: shower energy
    # boundary every 100 GeV
    fig6, axs6 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs6.flatten()

    for cat in range(n_en_cat):
        axs6.flatten()[cat].hist(showerEn_arr_cat_en_pi[cat], bins=50, range=(0,1100), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs6.flatten()[cat].hist(showerEn_arr_cat_en_pho[cat], bins=50, range=(0,1100), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower energy')
        axs6.flatten()[cat].set_ylabel('# trk')
        axs6.flatten()[cat].set_yscale('log')
        # add a box containing the energy range
        axs6.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs6.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'showerEn_en.png')) #save plot
    plt.close(fig6)

    # do plots in bins of energy: shower eta
    # boundary every 100 GeV
    fig6, axs6 = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs6.flatten()

    for cat in range(n_en_cat):
        axs6.flatten()[cat].hist(showerEta_arr_cat_en_pi[cat], bins=20, range=(1.5,3.1), density=norm, color='green', alpha=0.4, label=r'$\pi$')
        axs6.flatten()[cat].hist(showerEta_arr_cat_en_pho[cat], bins=20, range=(1.5,3.1), density=norm, color='orange', alpha=0.4, label=r'$\gamma$')
        axs6.flatten()[cat].legend()
        axs6.flatten()[cat].set_xlabel('shower eta')
        axs6.flatten()[cat].set_ylabel('# trk')
        axs6.flatten()[cat].set_yscale('log')
        # add a box containing the energy range
        axs6.flatten()[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs6.flatten()[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(out_dir, 'showerEta_en.png')) #save plot
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
