"""
plot LC multiplicity per category
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
from plotter_functions import _divide_en_categ

plt.style.use(hep.style.CMS)
mpl.use('agg')


# function to compute multiplicity in categories
def doMultiplicityPlots_cat(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str, n_layers: int = 48, n_en_cat : int =9) -> None:

    # define list of strings for energy bins
    en_bin_str = ['0 < E < 100 GeV', '100 < E < 200 GeV', '200 < E < 300 GeV', '300 < E < 400 GeV', '400 < E < 500 GeV', '500 < E < 600 GeV', '600 < E < 700 GeV', '700 < E < 800 GeV', 'E > 800 GeV']

    # --- PIONS ---
    # define array of arrays of arrays of multiplicity per layer (per each event)
    mult_matrix_pi = [[[] for _ in range(n_layers)] for _ in range(n_en_cat)]

    for i_file in data_list_pi:
        for i_evt in i_file:

            # create array of zeros of lenghts n_layer
            counter_arr_pi = np.zeros((n_layers,), dtype=int)

            # --- read 2D objects
            layerClusterMatrix_pi = i_evt.clus2d_feat.numpy()

            # --- read 3D objects
            trackster_pi = i_evt.clus3d_feat.numpy()

            # get the category number
            cat_number = _divide_en_categ(trackster_pi[2])

            # loop over the LC of the event
            for i_LC in range(layerClusterMatrix_pi.shape[0]):
                counter_arr_pi[int(layerClusterMatrix_pi[i_LC,5])] += 1

            # append the array of multiplicity per layer to the list of arrays
            for i_L in range(n_layers):
                mult_matrix_pi[cat_number][i_L].append(counter_arr_pi[i_L])

    # multiplicity matrix
    #print(mult_matrix_pi.shape)
    print(len(mult_matrix_pi))
    print(len(mult_matrix_pi[0]))
    print(len(mult_matrix_pi[0][0]))
    # pad the matrix such that the final shape is n_cat x n_layers x n_events
    # where n_events is the maximum number of events in a category
    # find the maximum lenght of the arrays in the matrix (maximum number of events in a category)
    max_len = -999
    for cat in range(n_en_cat):
        for layer in range(n_layers):
            #print(len(mult_matrix_pi[cat][layer]))
            #print(max_len)
            if len(mult_matrix_pi[cat][layer]) > max_len:
                max_len = len(mult_matrix_pi[cat][layer])

    print(max_len)

    # pad the matrix such that the final shape is n_cat x n_layers x n_events
    # where n_events is the maximum number of events in a category
    for cat in range(n_en_cat):
        for layer in range(n_layers):
            mult_matrix_pi[cat][layer] = np.pad(mult_matrix_pi[cat][layer], (0,max_len-len(mult_matrix_pi[cat][layer])), 'constant', constant_values=(0,0))

    mult_matrix_pi = np.array(mult_matrix_pi)
    print(mult_matrix_pi.shape)


    # define array that contains mean of multiplicity per layer per each category
    multi_arr_pi = np.mean(mult_matrix_pi, axis=2)
    print(multi_arr_pi.shape)
    #print(multi_arr_pi)
    # It's the same as doing the following:
    # mult_arr_pi = [[[] for _ in range(n_layers)] for _ in range(n_en_cat)]
    # for cat in range(n_en_cat):
    #     for layer in range(n_layers):
    #         mult_arr_pi[cat][layer] = mult_matrix_pi[cat][layer].mean()
    # mult_arr_pi = np.array(mult_arr_pi)
    # print(mult_arr_pi.shape)
    # print(mult_arr_pi)
    # and this is to check if the two numpy arrays are equal
    # True if two arrays have the same shape and elements, False otherwise.
    # print(np.array_equal(multi_arr_pi, mult_arr_pi))

    # define array that contains 95% quantile of multiplicity per layer per each category
    multi_arr_pi_95 = np.quantile(mult_matrix_pi, 0.95, axis=2)
    print(multi_arr_pi_95.shape)


    # --- PHOTONS ---
    # define array of arrays of arrays of multiplicity per layer (per each event)
    mult_matrix_pho = [[[] for _ in range(n_layers)] for _ in range(n_en_cat)]

    for i_file in data_list_pho:
        for i_evt in i_file:

            # create array of zeros of lenghts n_layer
            counter_arr_pho = np.zeros((n_layers,), dtype=int)

            # --- read 2D objects
            layerClusterMatrix_pho = i_evt.clus2d_feat.numpy()

            # --- read 3D objects
            trackster_pho = i_evt.clus3d_feat.numpy()

            # get the category number
            cat_number = _divide_en_categ(trackster_pho[2])

            # loop over the LC of the event
            for i_LC in range(layerClusterMatrix_pho.shape[0]):
                counter_arr_pho[int(layerClusterMatrix_pho[i_LC,5])] += 1

            # append the array of multiplicity per layer to the list of arrays
            for i_L in range(n_layers):
                mult_matrix_pho[cat_number][i_L].append(counter_arr_pho[i_L])

    # multiplicity matrix
    print(len(mult_matrix_pho))
    print(len(mult_matrix_pho[0]))
    print(len(mult_matrix_pho[0][0]))
    # pad the matrix such that the final shape is n_cat x n_layers x n_events
    # where n_events is the maximum number of events in a category
    # find the maximum lenght of the arrays in the matrix (maximum number of events in a category)
    max_len_pho = -999
    for cat in range(n_en_cat):
        for layer in range(n_layers):
            #print(len(mult_matrix_pho[cat][layer]))
            #print(max_len_pho)
            if len(mult_matrix_pho[cat][layer]) > max_len_pho:
                max_len_pho = len(mult_matrix_pho[cat][layer])
    print(max_len_pho)
    # pad the matrix such that the final shape is n_cat x n_layers x n_events
    # where n_events is the maximum number of events in a category
    for cat in range(n_en_cat):
        for layer in range(n_layers):
            mult_matrix_pho[cat][layer] = np.pad(mult_matrix_pho[cat][layer], (0,max_len_pho-len(mult_matrix_pho[cat][layer])), 'constant', constant_values=(0,0))
    mult_matrix_pho = np.array(mult_matrix_pho)
    print(mult_matrix_pho.shape)

    # define array that contains mean of multiplicity per layer per each category
    multi_arr_pho = np.mean(mult_matrix_pho, axis=2)
    print(multi_arr_pho.shape)

    # define array that contains 95% quantile of multiplicity per layer per each category
    multi_arr_pho_95 = np.quantile(mult_matrix_pho, 0.95, axis=2)
    print(multi_arr_pho_95.shape)


    # # plot multiplicity per layer per category
    # # a looot of plots, be careful
    # x_ticks = np.arange(0, 21, 2, dtype=int) # ticks for x axis - from 0 to 21 (excluded) with step 2
    # for cat in range(n_en_cat):
    #     for layer in range(n_layers):
    #         fig1, ax1 = plt.subplots(figsize=(12,10), tight_layout=True)
    #         ax1.hist(mult_matrix_pi[cat][layer], bins=20, range=(0,20), density=True, color='green', alpha=0.4, label=r'$\pi$')
    #         ax1.hist(mult_matrix_pho[cat][layer], bins=20, range=(0,20), density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    #         ax1.legend()
    #         ax1.set_xticks(x_ticks)
    #         ax1.set_xlabel('Multiplicity')
    #         ax1.set_ylabel('# LC')
    #         ax1.set_title('Layer '+str(layer)+' - '+en_bin_str[cat])
    #         plt.savefig(os.path.join(out_dir, 'mult_layer'+str(layer)+'_'+en_bin_str[cat]+'.png'))
    #         plt.close(fig1)

    # plot multiplicity mean per category
    fig, axs = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs.flatten()
    for cat in range(n_en_cat):
        axs.flatten()[cat].plot(np.arange(0, n_layers, 1), multi_arr_pi[cat], linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
        axs.flatten()[cat].plot(np.arange(0, n_layers, 1), multi_arr_pho[cat], linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
        axs.flatten()[cat].legend()
        axs.flatten()[cat].set_xlabel('Layer Number')
        axs.flatten()[cat].set_ylabel('Multiplicity mean')
        axs.flatten()[cat].set_title(en_bin_str[cat])
    plt.savefig(os.path.join(out_dir, 'mult_cat.png')) #save plot
    plt.close(fig)

    # plot multiplicity 95% quantile per category
    fig, axs = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs.flatten()
    for cat in range(n_en_cat):
        axs.flatten()[cat].plot(np.arange(0, n_layers, 1), multi_arr_pi_95[cat], linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
        axs.flatten()[cat].plot(np.arange(0, n_layers, 1), multi_arr_pho_95[cat], linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
        axs.flatten()[cat].legend()
        axs.flatten()[cat].set_xlabel('Layer Number')
        axs.flatten()[cat].set_ylabel('Multiplicity 95% quantile')
        axs.flatten()[cat].set_title(en_bin_str[cat])
    plt.savefig(os.path.join(out_dir, 'mult_95_cat.png')) #save plot
    plt.close(fig)
