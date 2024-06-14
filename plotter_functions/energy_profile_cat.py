"""
function to compute the energy profile of the trackster per category
"""

from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

from plotter_functions import _divide_en_categ, _divide_eta_categ


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
