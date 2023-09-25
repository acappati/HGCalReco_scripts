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

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import torch_geometric
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

plt.style.use(hep.style.CMS)




def openFiles(input_file_path, data_list):
    """
    function to open files
    """

    filenamelist = [ filename for filename in glob.glob( input_file_path + 'data_*.pt' )]
    for i in tqdm( filenamelist ):
        idx = torch.load(i)
        data_list.append(idx)



def doHisto(data_list_pho, data_list_pi, out_dir):
    """
    function to do histograms from the training samples
    """

    ### PHOTONS
    min_clusL_arr_pho = [] # array of all min_clusL
    max_clusL_arr_pho = [] # array of all max_clusL
    eta_arr_pho = [] # array of all eta

    # loop over files in data list
    for i_file_pho in data_list_pho:
        # loop over all events in one file
        for i_evt_pho in i_file_pho:

            # --- read 2D objects
            # layerClusterMatrix = matrix of all the LayerClusters in the file (number of rows) 
            #                      with their features (number of columns)
            # LayerCluster features: clusX,clusY,clusZ,clusE,clusT,clusL
            # there is one matrix of this kind for each event of the loadfile_pi
            layerClusterMatrix_pho = i_evt_pho.clus2d_feat


            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features: 
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster_pho = i_evt_pho.clus3d_feat.numpy() # transform the tensor in numpy array

            min_clusL_arr_pho.append(trackster_pho[4]) # fill array of all min cluster Layer
            max_clusL_arr_pho.append(trackster_pho[5]) # fill array of all max cluster Layer
            eta_arr_pho.append(trackster_pho[0]) # fill array of all eta

    
    ### PIONS
    min_clusL_arr_pi = [] # array of all min_clusL
    max_clusL_arr_pi = [] # array of all max_clusL
    eta_arr_pi = [] # array of all eta

    eta_min_pi = np.inf # min eta
    eta_max_pi = -np.inf # max eta

    # loop over files in data list
    for i_file_pi in data_list_pi:
        # loop over all events in one file
        for i_evt_pi in i_file_pi:

            # --- read 2D objects
            # layerClusterMatrix = matrix of all the LayerClusters in the file (number of rows) 
            #                      with their features (number of columns)
            # LayerCluster features: clusX,clusY,clusZ,clusE,clusT,clusL
            # there is one matrix of this kind for each event of the loadfile_pi
            layerClusterMatrix_pi = i_evt_pi.clus2d_feat


            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features: 
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster_pi = i_evt_pi.clus3d_feat.numpy() # transform the tensor in numpy array

            min_clusL_arr_pi.append(trackster_pi[4]) # fill array of all min cluster Layer
            max_clusL_arr_pi.append(trackster_pi[5]) # fill array of all max cluster Layer
            eta_arr_pi.append(trackster_pi[0]) # fill array of all eta

            #TODO decide eta bins
            #if trackster_pi[0] > 1.6 and trackster[0] < 1.8:
            #    min_clusL_arr_pi_1.append(trackster_pi[4]) 
            #elif ...

            #TODO plots in bins of pT

            #eta_min_pi = min(eta_min_pi, trackster_pi[0]) # min eta
            #eta_max_pi = max(eta_max_pi, trackster_pi[0]) # max eta


    # histos min-max L - inclusive
    fig, axs = plt.subplots(3, 2, figsize=(20,20), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, 47) # this way I have 48 bins from 0 to 47 : 48 bins = 48 layers 

    # hist of min_clusL photons
    axs[0][0].hist(min_clusL_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$') 
    axs[0][0].legend()        
    axs[0][0].set_xlabel('min clusL')
    axs[0][0].set_ylabel('# trk')

    # hist of min_clusL pions
    axs[1][0].hist(min_clusL_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs[1][0].legend()        
    axs[1][0].set_xlabel('min clusL')
    axs[1][0].set_ylabel('# trk')

    # hist of min_clusL both
    axs[2][0].hist(min_clusL_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs[2][0].hist(min_clusL_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[2][0].legend()        
    axs[2][0].set_xlabel('min clusL')
    axs[2][0].set_ylabel('# trk')

    # hist of max_clusL photons
    axs[0][1].hist(max_clusL_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[0][1].legend()        
    axs[0][1].set_xlabel('max clusL')
    axs[0][1].set_ylabel('# trk')

    # hist of max_clusL pions
    axs[1][1].hist(max_clusL_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs[1][1].legend()        
    axs[1][1].set_xlabel('max clusL')
    axs[1][1].set_ylabel('# trk')

    # hist of max_clusL both
    axs[2][1].hist(max_clusL_arr_pi, bins=binEdges_list, color='green', alpha=0.4, label=r'$\pi$')
    axs[2][1].hist(max_clusL_arr_pho, bins=binEdges_list, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[2][1].legend()        
    axs[2][1].set_xlabel('max clusL')
    axs[2][1].set_ylabel('# trk')
    
    plt.savefig(os.path.join(out_dir, 'minmaxL.png')) #save plot


    #TODO do plots in bins of eta
    # boundary between low and high density: 2.15 eta
    #fig, axs = plt.subplots(4, 2, figsize=(20,20), dpi=80, tight_layout=True)
    #binEdges_list = np.arange(0, 47) # this way I have 48 bins from 0 to 47 : 48 bins = 48 layers
    #etaBin_list_lowdensity = np.arange(1.5, 3.1, 0.2) # this way I have 8 bins from 1.5 to 3.1 

    

def doENprofile(data_list_pho, out_dir):
    """
    function to compute the energy profile
    """
    ### PHOTONS
    # energy array for each LC for photons
    # create an array of 48 empty arrays
    en_arr_pho = [[] for i in range(48)]

    # create an array of 48 entries for the energy mean for each LC
    en_mean_arr_pho = [0 for i in range(48)]

    # create an array of 48 entries for the energy median for each LC
    en_median_arr_pho = [0 for i in range(48)]
    
    # loop over files in data list
    for i_file_pho in data_list_pho:
        # loop over all events in one file
        for i_evt_pho in i_file_pho:

            # --- read 2D objects
            # layerClusterMatrix = matrix of all the LayerClusters in the file (number of rows) 
            #                      with their features (number of columns)
            # LayerCluster features: clusX,clusY,clusZ,clusE,clusT,clusL
            # there is one matrix of this kind for each event of the loadfile_pi
            layerClusterMatrix_pho = i_evt_pho.clus2d_feat.numpy() # transform the tensor in numpy array

            # need to read the energy of each LayerCluster
            # the energy is in the 4th column of the matrix layerClusterMatrix_pho [:,3] 
            # need to cycle over all the rows of the matrix
            # then fill an array with the energy of each layer
            # and then compute the mean or the median of the array
            # and plot the energy profile
            # do the same for pions
            for i_LC in range(len(layerClusterMatrix_pho)): #loop over matrix rows
                en_arr_pho[int(layerClusterMatrix_pho[i_LC,5])].append(layerClusterMatrix_pho[i_LC,3]) # fill array of all energies of all LCs in all events
            
    # compute the mean and median of the energy array for each LC
    for i in range(len(en_arr_pho)):
        # compute the mean if the array is not empty
        en_mean_arr_pho[i] = np.mean(en_arr_pho[i] if len(en_arr_pho[i]) > 0 else [0])
        # compute the median if the array is not empty  
        en_median_arr_pho[i] = np.median(en_arr_pho[i] if len(en_arr_pho[i]) > 0 else [0])

    #TODO plot heatmap of energy of layer clusters per layer number (pad with zeros if not same length)

    
    ### PIONS
    # energy array for each LC for photons
    # create an array of 48 empty arrays
    en_arr_pi = [[] for i in range(48)]

    # create an array of 48 entries for the energy mean for each LC
    en_mean_arr_pi = [0 for i in range(48)]

    # create an array of 48 entries for the energy median for each LC
    en_median_arr_pi = [0 for i in range(48)]
    
    # loop over files in data list
    for i_file_pi in data_list_pi:
        # loop over all events in one file
        for i_evt_pi in i_file_pi:

            layerClusterMatrix_pi = i_evt_pi.clus2d_feat.numpy() # transform the tensor in numpy array

            for i_LC in range(len(layerClusterMatrix_pi)):
                en_arr_pi[int(layerClusterMatrix_pi[i_LC,5])].append(layerClusterMatrix_pi[i_LC,3]) # fill array of all energies of all LCs in all events
            
    # compute the mean and median of the energy array for each LC
    for i in range(len(en_arr_pi)):
        # compute the mean if the array is not empty
        en_mean_arr_pi[i] = np.mean(en_arr_pi[i] if len(en_arr_pi[i]) > 0 else [0])
        # compute the median if the array is not empty  
        en_median_arr_pi[i] = np.median(en_arr_pi[i] if len(en_arr_pi[i]) > 0 else [0])
        

    # plot the energy profile
    fig, axs = plt.subplots(1, 2, figsize=(20,10), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, 48) # this way I have 49 bins from 0 to 48 : 48 bins = 48 layers   

    axs[0].plot(binEdges_list, en_mean_arr_pho, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[0].plot(binEdges_list, en_mean_arr_pi, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    axs[0].legend()
    axs[0].set_xlabel('LayerCluster')
    axs[0].set_ylabel('Energy mean')

    axs[1].plot(binEdges_list, en_median_arr_pho, linewidth=4, color='orange', alpha=0.4, label=r'$\gamma$')
    axs[1].plot(binEdges_list, en_median_arr_pi, linewidth=4, color='green', alpha=0.4, label=r'$\pi$')
    axs[1].legend()
    axs[1].set_xlabel('LayerCluster')
    axs[1].set_ylabel('Energy median')

    plt.savefig(os.path.join(out_dir, 'en_profile.png')) #save plot
    






if __name__ == "__main__" :

    ## output directory
    today = date.today()
    print('creating output dir...')
    out_dir = str(today)+'_plots'
    os.makedirs(out_dir,exist_ok=True) #check if output dir exist

    ## input files photons
    inpath_pho = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_DATA/ntup_pho_frac0p8/'
    data_list_pho = []
    print('loading photon files...')
    openFiles(inpath_pho, data_list_pho)

    ## input files pions
    inpath_pi = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_DATA/ntup_pi_frac0p8/'
    data_list_pi = []
    print('loading pions files...')
    openFiles(inpath_pi, data_list_pi)

    ## plots
    print('doing plots...')
    doHisto(data_list_pho, data_list_pi, out_dir)

    doENprofile(data_list_pho, out_dir)