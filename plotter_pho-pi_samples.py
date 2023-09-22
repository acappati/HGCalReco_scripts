#!/usr/bin/env/python3

## ---
#  script to plot variables from training samples
#  run with: python3 plotter_pho-pi_samples.py
## ---

import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import mplhep as hep
import os
import optparse
import os.path as osp
import math
import torch_geometric
import torch
import sys
from tqdm import tqdm as tqdm
#import pandas as pd
from torch_geometric.data import Data
import glob
from datetime import date

plt.style.use(hep.style.CMS)



def openFiles(input_file_path, data_list):
    
    filenamelist = [ filename for filename in glob.glob( input_file_path + 'data_*.pt' )]
    for i in tqdm( filenamelist ):
        idx = torch.load(i)
        data_list.append(idx)



def doPlots(data_list_pho, data_list_pi, out_dir):

    ### PHOTONS
    min_clusL_arr_pho = [] # array of all min_clusL
    max_clusL_arr_pho = [] # array of all max_clusL

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

    
    ### PIONS
    min_clusL_arr_pi = [] # array of all min_clusL
    max_clusL_arr_pi = [] # array of all max_clusL

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


    # histos min-max L
    fig, axs = plt.subplots(3, 2, figsize=(20,20), dpi=80, tight_layout=True)
    binEdges_list = np.arange(0, 48) # this way I have 49 bins from 0 to 48 : 48 bins = 48 layers 

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
    doPlots(data_list_pho, data_list_pi, out_dir)
