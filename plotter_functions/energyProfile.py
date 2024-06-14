"""
function to compute the energy profile of the trackster
"""

from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm


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
