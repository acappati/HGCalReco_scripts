"""
function to check events with shower extensions with small values
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

# import modules
from plotter_functions import _divide_en_categ

plt.style.use(hep.style.CMS)
mpl.use('agg')


def findMinima(data_list, out_dir, norm=True, n_en_cat : int =9) -> list :
    """
    find minima in the shower extension distribution
    return the minimum between the main peak of the shower and the low energy deposit
    """

    # define array of min_clusL and max_clusL for each pT bin
    min_clusL_arr_cat_en = [[] for i in range(n_en_cat)]
    max_clusL_arr_cat_en = [[] for i in range(n_en_cat)]
    extShower_arr_cat_en = [[] for i in range(n_en_cat)]
    showerEn_arr_cat_en  = [[] for i in range(n_en_cat)]
    showerEta_arr_cat_en = [[] for i in range(n_en_cat)]


    # define array of strings for energy bins
    en_bin_str = ['0 < E < 100 GeV', '100 < E < 200 GeV', '200 < E < 300 GeV', '300 < E < 400 GeV', '400 < E < 500 GeV', '500 < E < 600 GeV', '600 < E < 700 GeV', '700 < E < 800 GeV', 'E > 800 GeV']


    # loop over files in data list
    for i_file in data_list:
        # loop over all events in one file
        for i_evt in i_file:

            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features:
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster = i_evt.clus3d_feat.numpy() # transform the tensor in numpy array

            # get the energy category number
            cat_en_n = _divide_en_categ(trackster[2])
            # divide in energy bins
            min_clusL_arr_cat_en[cat_en_n].append(trackster[4])
            max_clusL_arr_cat_en[cat_en_n].append(trackster[5])
            extShower_arr_cat_en[cat_en_n].append(abs(trackster[5]-trackster[4]))
            showerEn_arr_cat_en[cat_en_n].append(trackster[2])
            showerEta_arr_cat_en[cat_en_n].append(trackster[0])



    binEdges_list = np.arange(0, 48, 1) # this way I have 48 bins from 0 to 47 : 48 bins = 48 layers
    minima_list = [10., 17., 18., 19., 20., 20., 20., 20., 21.] # chosen by eye

    # do plots in bins of energy: shower extension
    # boundary every 100 GeV
    fig, axs = plt.subplots(3, 3, figsize=(20,20), dpi=80, tight_layout=True)
    axs = axs.flatten()

    for cat in range(n_en_cat):

        counts, edges = np.histogram(extShower_arr_cat_en[cat], bins=binEdges_list, range=(0,48), density=norm)

        axs[cat].stairs(counts, edges, color='orange', label=r'$\gamma$')
        axs[cat].legend()
        axs[cat].set_xlabel('shower extension')
        axs[cat].set_ylabel('# trk')
        # add a box containing the energy range
        axs[cat].text(0.6, 0.6, en_bin_str[cat], transform=axs[cat].transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # # convolve the counts to smooth the histo
        # kernel = np.ones(3)/3
        # smooth = np.convolve(counts, kernel, mode='same')
        # # find the relative minima
        # minima = argrelmin(smooth[3:])[0]
        # idx_min = minima[0]

        # # try: find peak, se trova due picchi, fai findrelmin tra i due massimi
        # peaks = find_peaks(counts)
        # if len(peaks[0])>1:
        #     minima = argrelmin(counts[peaks[0][0]:peaks[0][1]])[0]
        #     idx_min = minima[0]+peaks[0][0]
        # else:
        #     minima = argrelmin(counts)[0]
        #     idx_min = minima[0]

        # # kernel density estimation
        # kde = gaussian_kde(extShower_arr_cat_en[cat])
        # x = np.linspace(0, 48, 1000)
        # y = kde(x)

        # ## find index of the first non trivial min of the counts
        # grad = np.gradient(y, x)
        # hess = np.gradient(grad, x)
        # idx_min = np.where((grad==0) & (hess>0))[0][0]

        ## find index of the first non trivial min of the counts
        # exclude the first 3 bins to avoid to take 0 as minimum
        #peaks = find_peaks(max(counts)-counts)

        idx_min = minima_list[cat]
        # add vertical line at the minimum
        axs[cat].axvline(x=idx_min, color='red', linestyle='--', label='min')

    plt.savefig(os.path.join(out_dir, 'check_extShower_en.png')) #save plot
    plt.close(fig)


    return minima_list



def checkShowerExt(data_list, out_dir, norm=True) -> None :
    """
    Check events with shower extensions with small values
    """

    # create out dir for plots of shower extension
    out_dir_showerExt = out_dir+'/'+'checkShowerExt'
    os.makedirs(out_dir_showerExt, exist_ok=True) #check if output dir exist

    # define array of strings for energy bins
    en_bin_str = ['0 < E < 100 GeV', '100 < E < 200 GeV', '200 < E < 300 GeV', '300 < E < 400 GeV', '400 < E < 500 GeV', '500 < E < 600 GeV', '600 < E < 700 GeV', '700 < E < 800 GeV', 'E > 800 GeV']

    # find minima per category
    minima_list = findMinima(data_list, out_dir, norm=norm)
    print('minima list filled')
    print(minima_list)

    # counter to keep count of short showers
    i_short_shower = 0

    # loop over files in data list
    for i_file in data_list:
        # loop over all events in one file
        for i_evt in i_file:

            # --- read 3D objects
            # the clus3d_feat is a tensor of only 6 features:
            # trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL)
            trackster = i_evt.clus3d_feat.numpy() # transform the tensor in numpy array

            # get the energy category number
            cat_en_n = _divide_en_categ(trackster[2])

            # check if the shower extension is below the minimum
            if abs(trackster[5]-trackster[4]) < minima_list[cat_en_n]:
                print('Event with small shower extension')
                print('trackster quantities:')
                print('eta:', trackster[0])
                print('phi:', trackster[1])
                print('energy:', trackster[2])
                print('time:', trackster[3])
                print('min(clusL):', trackster[4])
                print('max(clusL):', trackster[5])
                print('shower extension:', abs(trackster[5]-trackster[4]))
                print('')

                # --- read 2D objects
                layerClusterMatrix = i_evt.clus2d_feat.numpy()

                # plot 2D clusters
                fig, axs = plt.subplots(1, 3, figsize=(20,12),dpi=80)

                # plot the scatter plot of the LCs in the events
                # --- Layer number vs x scatter plot
                axs[0].scatter(layerClusterMatrix[:,0],layerClusterMatrix[:,5],s=1000.*layerClusterMatrix[:,3], color='orange', alpha=0.4)
                axs[0].set_xlabel('x (cm)')
                axs[0].set_ylabel('Layer number')
                # --- Layer number vs y scatter plot
                axs[1].scatter(layerClusterMatrix[:,1],layerClusterMatrix[:,5],s=1000.*layerClusterMatrix[:,3], color='orange', alpha=0.4)
                axs[1].set_xlabel('y (cm)')
                axs[1].set_ylabel('Layer number')
                # --- y vs x scatter plot
                axs[2].scatter(layerClusterMatrix[:,0],layerClusterMatrix[:,1],s=1000.*layerClusterMatrix[:,3], color='orange', alpha=0.4)
                axs[2].set_xlabel('x (cm)')
                axs[2].set_ylabel('y (cm)')

                # add category info
                my_box = dict(boxstyle='round', facecolor='white', alpha=0.5)
                axs[2].text(1.04, 0.9, en_bin_str[cat_en_n], transform=axs[2].transAxes, fontsize=16, verticalalignment='top', bbox=my_box)

                # add pad with trackster info
                my_text = 'Trackster quantities: \n'
                my_text += 'eta: '+str(trackster[0])+'\n'
                my_text += 'phi: '+str(trackster[1])+'\n'
                my_text += 'energy: '+str(trackster[2])+'\n'
                my_text += 'time: '+str(trackster[3])+'\n'
                my_text += 'min(clusL): '+str(trackster[4])+'\n'
                my_text += 'max(clusL): '+str(trackster[5])+'\n'
                my_text += 'shower extension: '+str(abs(trackster[5]-trackster[4]))
                axs[2].text(1.04, 0.8, my_text, transform=axs[2].transAxes, fontsize=16, verticalalignment='top', bbox=my_box)


                plt.tight_layout()
                plotname = 'ShowerExt_'+str(i_short_shower)+'.png'
                plt.savefig(os.path.join(out_dir_showerExt, plotname)) #save plot
                plt.close(fig)

                print('plot done for event:', i_short_shower)
                print('')

                i_short_shower += 1 # increment counter for short showers
