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

# import modules
from plotter_functions import (_divide_en_categ, _divide_eta_categ, doHisto,
                               openFiles)

plt.style.use(hep.style.CMS)
mpl.use('agg')















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
    print(len(en_arr_frac_matrix))
    print(len(en_arr_frac_matrix[0]))
    print(len(en_arr_frac_matrix[0][0]))
    # the matrix has dimension n_cat x n_events x n_layers
    # find the max lenght of the arrays in the list of events
    max_n_events = max([len(i) for i in en_arr_frac_matrix]) # maximum number of events in a category
    print(max_n_events)
    # which is the same as doing the following
    # max_n_evt = -999
    # for cat in range(n_cat):
    #     if len(en_arr_frac_matrix[cat]) > max_n_evt:
    #         max_n_evt = len(en_arr_frac_matrix[cat])
    # print(max_n_evt)

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
    binEdges_list = np.arange(0, n_layers, 1) #list of 48 elements, from 0 to 47
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
    ax1.hist(fracH_arr_pi, bins=50, range=(0.,0.21), density=True, color='green', alpha=0.4, label=r'$\pi$')
    ax1.hist(fracH_arr_pho, bins=50, range=(0.,0.21), density=True, color='orange', alpha=0.4, label=r'$\gamma$')
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

    # plot gun features
    fig, axs = plt.subplots(3, 2, figsize=(20,20), dpi=80, tight_layout=True)
    axs.flatten()
    # plot eta
    axs.flatten()[0].hist(gun_matrix_pi[:,0], bins=50, range=(1.2,3.2),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[0].hist(gun_matrix_pho[:,0], bins=50, range=(1.2,3.2),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[0].legend()
    axs.flatten()[0].set_xlabel('eta')
    axs.flatten()[0].set_ylabel('# trk')
    # plot phi
    axs.flatten()[1].hist(gun_matrix_pi[:,1], bins=50, range=(-4.,4.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[1].hist(gun_matrix_pho[:,1], bins=50, range=(-4.,4.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[1].legend()
    axs.flatten()[1].set_xlabel('phi')
    axs.flatten()[1].set_ylabel('# trk')
    # plot energy in EM part of HGCal
    axs.flatten()[2].hist(gun_matrix_pi[:,2], bins=50, range=(0.,1200.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[2].hist(gun_matrix_pho[:,2], bins=50, range=(0.,1200.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[2].legend()
    axs.flatten()[2].set_xlabel('energy in CEE')
    axs.flatten()[2].set_ylabel('# trk')
    # plot total energy
    axs.flatten()[3].hist(gun_matrix_pi[:,3], bins=50, range=(0.,1200.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[3].hist(gun_matrix_pho[:,3], bins=50, range=(0.,1200.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[3].legend()
    axs.flatten()[3].set_xlabel('total energy')
    axs.flatten()[3].set_ylabel('# trk')
    # plot ratio
    axs.flatten()[4].hist(gun_matrix_pi[:,4], bins=50, range=(0.,5.),
                density=True, color='green', alpha=0.4, label=r'$\pi$')
    axs.flatten()[4].hist(gun_matrix_pho[:,4], bins=50, range=(0.,5.),
                density=True, color='orange', alpha=0.4, label=r'$\gamma$')
    axs.flatten()[4].legend()
    axs.flatten()[4].set_xlabel('ratio')
    axs.flatten()[4].set_ylabel('# trk')


    plt.savefig(os.path.join(out_dir, 'gunFeats.png')) #save plot
    plt.close(fig)


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
        plt.savefig(os.path.join(out_dir, 'mult_layer'+str(i_L)+'.png')) #save plot
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



if __name__ == "__main__" :

    ## output directory
    today = date.today()
    print('Creating output dir...')
    out_dir = str(today)+'_plots'
    os.makedirs(out_dir, exist_ok=True) #check if output dir exist

    ## input files photons
    inpath_pho = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/ntup_pho_21052024/'
    data_list_pho = openFiles(inpath_pho, desc='Loading photon files')

    ## input files pions
    inpath_pi = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/ntup_pi_21052024/'
    data_list_pi = openFiles(inpath_pi, desc='Loading pions files')

    ## plots
    print('doing plots...')

    print('doing histos...')
    doHisto(data_list_pho, data_list_pi, out_dir, False)

    #doENprofile(data_list_pho, data_list_pi, out_dir)

    #doGunPlots(data_list_pho, data_list_pi, out_dir)

    #doMultiplicityPlots(data_list_pho, data_list_pi, out_dir)

    #doMultiplicityPlots_cat(data_list_pho, data_list_pi, out_dir)

    #doVisualizationPlots(data_list_pho, data_list_pi, out_dir)
