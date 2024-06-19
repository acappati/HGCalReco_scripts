"""
plot LC multiplicity
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

plt.style.use(hep.style.CMS)
mpl.use('agg')


# function to plot LC multiplicity
# we want the average multiplicity of LC per layer
def doMultiplicityPlots(data_list_pho: List[Data], data_list_pi: List[Data], out_dir: str, n_layers: int = 48) -> None:

    # --- pions
    mult_matrix_pi = [[] for _ in range(n_layers)] # array of arrays of multiplicity per layer (per each event)

    for i_file in data_list_pi:
        for i_evt in i_file:

            # create array for each Layer
            counter_arr_pi = np.zeros((n_layers,), dtype=int) # array of zeros of length n_layers (one element per layer)
            # shape of the array: (48,) - array of 48 components
            # same thing of the following but faster
            # counter_arr_pi = [0 for _ in range(n_layers)]

            # --- read 2D objects
            # layerClusterMatrix = matrix of all the LayerClusters in the file
            # LayerCluster features: clusX,clusY,clusZ,clusE,clusT,clusL
            # (number of rows)
            # with their features (number of columns)
            # there is one matrix of this kind for each event of the loadfile_pi
            layerClusterMatrix_pi = i_evt.clus2d_feat.numpy()

            # loop over the LC of the event
            # shape of the matrix: (number of LC, 6)
            # number of LC = number of rows -> cycle on shape[0]
            for i_LC in range(layerClusterMatrix_pi.shape[0]):
                counter_arr_pi[int(layerClusterMatrix_pi[i_LC,5])] += 1 # count how many LC per layer; there is one element per layer

            # append the array of multiplicity per layer to the list of arrays
            for i_L in range(n_layers):
                mult_matrix_pi[i_L].append(counter_arr_pi[i_L])

    # multiplicity matrix
    mult_matrix_pi = np.array(mult_matrix_pi)
    print(mult_matrix_pi.shape)

    # array with mean per layer of the multiplicity
    # define array that contains mean of multiplicity per layer
    # do the mean per each layer, meaning per each row of the matrix
    # axis=1 means that we take the mean of each row
    mult_arr_pi = np.mean(mult_matrix_pi, axis=1)
    # It would be the same as doing the following, but it's faster
    # mult_arr_pi = [] # array of multiplicity per layer
    # for i_L in range(n_layers):
    #     mult_arr_pi.append(np.mean(mult_matrix_pi[i_L]))
    #print(mult_arr_pi)

    # array with 95% quantile per layer of the multiplicity
    # define array that contains 95% quantile of multiplicity per layer
    # do the 95% quantile per each layer, meaning per each row of the matrix
    # axis=1 means that we take the 95% quantile of each row
    mult_arr_pi_95 = np.quantile(mult_matrix_pi, 0.95, axis=1)
    # It would be the same as doing the following, but it's faster
    # mult_arr_pi_95 = []
    # for i_L in range(n_layers):
    #     mult_arr_pi_95.append(np.quantile(mult_matrix_pi[i_L], 0.95))
    # print(mult_arr_pi_95)


    # --- photons
    mult_matrix_pho = [[] for _ in range(n_layers)] # array of arrays of multiplicity per layer (per each event)

    for i_file in data_list_pho:
        for i_evt in i_file:

            # create array for each Layer
            counter_arr_pho = np.zeros((n_layers,), dtype=int) # array of zeros of length n_layers (one element per layer)

            # --- read 2D objects
            layerClusterMatrix_pho = i_evt.clus2d_feat.numpy()

            # loop over the LC of the event
            for i_LC in range(layerClusterMatrix_pho.shape[0]):
                counter_arr_pho[int(layerClusterMatrix_pho[i_LC,5])] += 1 # count how many LC per layer; there is one element per layer

            # append the array of multiplicity per layer to the list of arrays
            for i_L in range(n_layers):
                mult_matrix_pho[i_L].append(counter_arr_pho[i_L])

    # multiplicity matrix
    mult_matrix_pho = np.array(mult_matrix_pho)
    print(mult_matrix_pho.shape)

    # array with mean per layer of the multiplicity
    mult_arr_pho = np.mean(mult_matrix_pho, axis=1)
    #print(mult_arr_pho)

    # array with 95% quantile per layer of the multiplicity
    mult_arr_pho_95 = np.quantile(mult_matrix_pho, 0.95, axis=1)
    #print(mult_arr_pho_95)

    # plot multiplicity per layer
    # create out dir for plots of multiplicity per layer
    out_dir_mult_layer = out_dir+'/'+'mult_layer'
    os.makedirs(out_dir_mult_layer, exist_ok=True) #check if output dir exist

    x_ticks = np.arange(0, 21, 2, dtype=int) # ticks for x axis - from 0 to 21 (excluded) with step 2
    for i_L in range(n_layers):
        fig1, ax1 = plt.subplots(figsize=(12,10), tight_layout=True)
        ax1.hist(mult_matrix_pi[i_L], bins=20, range=(0,20), density=True, color='green', alpha=0.4, label=r'$\pi$')
        ax1.hist(mult_matrix_pho[i_L], bins=20, range=(0,20), density=True, color='orange', alpha=0.4, label=r'$\gamma$')
        ax1.legend()
        ax1.set_xticks(x_ticks)
        ax1.set_xlabel('Multiplicity')
        ax1.set_ylabel('# LC')
        ax1.set_title('Layer '+str(i_L))
        plt.savefig(os.path.join(out_dir_mult_layer, 'mult_layer'+str(i_L)+'.png')) #save plot
        plt.close(fig1)


    # plot multiplicity mean
    fig, ax = plt.subplots(figsize=(16,10), dpi=80, tight_layout=True)
    layer_list = np.arange(0, n_layers, 1)
    ax.plot(layer_list, mult_arr_pi, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    ax.plot(layer_list, mult_arr_pho, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    ax.legend()
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Multiplicity mean')
    plt.savefig(os.path.join(out_dir, 'mult.png')) #save plot
    plt.close(fig)

    # plot multiplicity 95% quantile
    fig, ax = plt.subplots(figsize=(16,10), dpi=80, tight_layout=True)
    layer_list = np.arange(0, n_layers, 1)
    ax.plot(layer_list, mult_arr_pi_95, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    ax.plot(layer_list, mult_arr_pho_95, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    ax.legend()
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Multiplicity 95% quantile')
    plt.savefig(os.path.join(out_dir, 'mult_95.png')) #save plot
    plt.close(fig)
