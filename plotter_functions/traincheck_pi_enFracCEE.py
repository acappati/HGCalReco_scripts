"""
function to check pions with enough energy fraction in CEE
and to decide cuts to apply to training samples (to make pions resemble photons)
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from tqdm import tqdm as tqdm

plt.style.use(hep.style.CMS)
mpl.use('agg')


def traincheck_enFracCEE(data_list) -> list:
    """
    function to select pions with enough energy fraction CEE
    """

    # define the output list containing the selected events
    # list of tensors
    output_tensor_list = []

    # loop over files in data list
    for i_file in data_list:
        # loop over all events in one file
        for i_evt in i_file:

            # --- read 2D objects
            # properties: clusX,clusY,clusZ,clusE,clusT,clusL
            LCmatrix = i_evt.clus2d_feat.numpy()


            # --- select only events with 70% energy in the CEE
            # consider CEE limit: layer 27 is the first layer of CEH
            # if layer numbering starts from 1

            # here define a mask to define the fraction of energy in CEE:
            # - LCmatrix[:,5] < 27: select only LCs in CEE
            # - np.sum(LCmatrix[LCmatrix[:,5]< 27][:,3]): sum of energy of LCs in CEE
            # - np.sum(LCmatrix[:,3]): sum of energy of all LCs
            fracCEE = np.sum(LCmatrix[LCmatrix[:,5]< 27][:,3])/np.sum(LCmatrix[:,3])

            # then require to have at least 70% of energy in CEE
            if fracCEE < 0.7:
                continue

            # if the event passes the selection, append the it to the output list
            output_tensor_list.append(LCmatrix)

    return output_tensor_list



def plot_enFracCEE(data_list, out_dir) -> None:
    """
    function to check pions with enought energy in CEE
    and do plots
    """

    # read selected data
    # events with energy fraction in CEE >70%
    print('[traincheck_enFrac] Selecting events with energy fraction in CEE >70%')
    LCtensor_list = traincheck_enFracCEE(data_list)


    print('[traincheck_enFrac] Plotting events')

    # plot 2D clusters
    fig, axs = plt.subplots(1, 3, figsize=(20,12),dpi=80)


    # loop over events saved in the list of tensors
    for iLCtensor in range(0,100): #len(LCtensor_list)

        # plot the scatter plot of the LCs in the events
        # --- Layer number vs x scatter plot
        axs[0].scatter(LCtensor_list[iLCtensor][:,0],LCtensor_list[iLCtensor][:,5],s=LCtensor_list[iLCtensor][:,3], alpha=0.4)
        axs[0].set_xlabel('x (cm)')
        axs[0].set_ylabel('Layer number')
        # --- Layer number vs y scatter plot
        axs[1].scatter(LCtensor_list[iLCtensor][:,1],LCtensor_list[iLCtensor][:,5],s=LCtensor_list[iLCtensor][:,3], alpha=0.4)
        axs[1].set_xlabel('y (cm)')
        axs[1].set_ylabel('Layer number')
        # --- y vs x scatter plot
        axs[2].scatter(LCtensor_list[iLCtensor][:,0],LCtensor_list[iLCtensor][:,1],s=LCtensor_list[iLCtensor][:,3], alpha=0.4)
        axs[2].set_xlabel('x (cm)')
        axs[2].set_ylabel('y (cm)')


    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'traincheck_pi_fracCEE.png')) #save plot
    plt.close(fig)
