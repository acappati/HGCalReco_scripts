"""
function to check events with a large fraction of energy in the hadronic part of the calorimeter
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.signal import argrelmin, find_peaks
from scipy.stats import gaussian_kde
#import pandas as pd
from tqdm import tqdm as tqdm

plt.style.use(hep.style.CMS)
mpl.use('agg')


def checkFractionCEH(data_list, out_dir, color_histo):
    """
    function to check events with a large fraction of energy in the hadronic part of the calorimeter
    """

    # create out dir for plots
    if color_histo == 'green':
        path_appendix = 'checkCEHfraction_pi'
    elif color_histo == 'orange':
        path_appendix = 'checkCEHfraction_pho'
    out_dir_ceh = out_dir+'/'+path_appendix
    os.makedirs(out_dir_ceh, exist_ok=True) #check if output dir exist

    # check events with fraction of energy in the hadronic part of the calorimeter above a certain limit
    limit_value = 0.1

    # counter to keep count of large fraction in ceh
    i_frac_ceh = 0

    # loop over files in data list
    for i_file in data_list:
        # loop over all events in one file
        for i_evt in i_file:

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

            # --- read gun properties
            gun = i_evt.gun_feat.numpy() # transform the tensor in numpy array


            ## compute the energy fraction in the hadronic part of HGCal
            # consider the last 22 layers
            fracH = np.sum(layerClusterMatrix[layerClusterMatrix[:,5]> 26][:,3])/np.sum(layerClusterMatrix[:,3])

            #print('fraction of energy in the hadronic part:', fracH)

            # check if the fraction of energy in the hadronic part is above the limit
            if fracH > limit_value:

                print('Event with large fraction of energy in the hadronic part')
                print('fraction of energy in the hadronic part:', fracH)
                print('trackster quantities:')
                print('eta:', trackster[0])
                print('phi:', trackster[1])
                print('energy:', trackster[2])
                print('time:', trackster[3])
                print('min(clusL):', trackster[4])
                print('max(clusL):', trackster[5])
                print('')
                print('LayerCluster quantities:')
                print('sum of LC energy:', np.sum(layerClusterMatrix[:,3]))
                print('')
                print('gun quantities:')
                print('eta:', gun[0])
                print('phi:', gun[1])
                print('energy:', gun[2])
                print('tot energy:', gun[3])
                print('ratio:', gun[4])



                # plot 2D clusters
                fig, axs = plt.subplots(1, 3, figsize=(20,12),dpi=80)

                # plot the scatter plot of the LCs in the events
                # --- Layer number vs x scatter plot
                axs[0].scatter(layerClusterMatrix[:,0],layerClusterMatrix[:,5],s=1000.*layerClusterMatrix[:,3], color=color_histo, alpha=0.4)
                axs[0].set_xlabel('x (cm)')
                axs[0].set_ylabel('Layer number')
                # --- Layer number vs y scatter plot
                axs[1].scatter(layerClusterMatrix[:,1],layerClusterMatrix[:,5],s=1000*layerClusterMatrix[:,3], color=color_histo, alpha=0.4)
                axs[1].set_xlabel('y (cm)')
                axs[1].set_ylabel('Layer number')
                # --- y vs x scatter plot
                axs[2].scatter(layerClusterMatrix[:,0],layerClusterMatrix[:,1],s=1000*layerClusterMatrix[:,3], color=color_histo, alpha=0.4)
                axs[2].set_xlabel('x (cm)')
                axs[2].set_ylabel('y (cm)')

                # add pad with trackster info
                my_box = dict(boxstyle='round', facecolor='white', alpha=0.5)
                my_text = 'FracH: '+str(fracH)+'\n'
                my_text += '\n'
                my_text += 'Trackster quantities: \n'
                my_text += 'eta: '+str(trackster[0])+'\n'
                my_text += 'phi: '+str(trackster[1])+'\n'
                my_text += 'energy: '+str(trackster[2])+'\n'
                my_text += 'time: '+str(trackster[3])+'\n'
                my_text += 'min(clusL): '+str(trackster[4])+'\n'
                my_text += 'max(clusL): '+str(trackster[5])+'\n'
                my_text += 'shower extension: '+str(abs(trackster[5]-trackster[4]))+'\n'
                my_text += '\n'
                my_text += 'LayerCluster quantities: \n'
                my_text += 'sum of LC energy: '+str(np.sum(layerClusterMatrix[:,3]))+'\n'
                my_text += '\n'
                my_text += 'Gun quantities: \n'
                my_text += 'eta: '+str(gun[0])+'\n'
                my_text += 'phi: '+str(gun[1])+'\n'
                my_text += 'energy: '+str(gun[2])+'\n'
                my_text += 'tot energy: '+str(gun[3])+'\n'
                my_text += 'ratio: '+str(gun[4])


                axs[2].text(1.04, 0.8, my_text, transform=axs[2].transAxes, fontsize=16, verticalalignment='top', bbox=my_box)


                plt.tight_layout()
                plotname = 'fractionCEH_'+str(i_frac_ceh)+'.png'
                plt.savefig(os.path.join(out_dir_ceh, plotname)) #save plot
                plt.close(fig)

                print('plot done for event:', i_frac_ceh)
                print('')

                i_frac_ceh += 1 # increment counter for large fraction in ceh
