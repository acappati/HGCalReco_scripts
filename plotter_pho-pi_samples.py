#!/usr/bin/env/python3

## ---
#  script to plot variables from training samples
#  run with: python3 plotter_pho-pi_samples.py
## ---

import os
from datetime import date

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm as tqdm

from plotter_functions import *
from tools import *

plt.style.use(hep.style.CMS)
mpl.use('agg')





if __name__ == "__main__" :

    ## output directory
    today = date.today()
    print('Creating output dir...')
    out_dir = str(today)+'_plots'
    os.makedirs(out_dir, exist_ok=True) #check if output dir exist


    ### new samples
    #### gun_feat : Gun properties
    # - cpeta = calo particle eta
    # - cpphi = calo particle phi
    # - cpen  = calo particle energy in the EM part of HGCal
    # - cpent = calo particle total energy
    # - rat   = ratio between EM part of the trackster and the energy of the calo particle (which is already in the EM part)

    #### clus2d_feat - clusX,clusY,clusZ,clusE,clusT,clusL : Layer Cluster properties
    # - position_x
    # - position y
    # - position_z
    # - energy
    # - cluster_time
    # - cluster_layer_id

    #### clus3d_feat - trkcluseta,trkclusphi,trkclusen,trkclustime, min(clusL),max(clusL) : trackster eta,phi,energy,time,min_layer,max_layer
    # - eta
    # - phi
    # - energy
    # - time
    # - min cluster layer
    # - max cluster layer



    ## input files photons
    # inpath_pho = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/ntup_pho_18082024/'
    inpath_pho = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/S2Rmin_pho2to15_27092024/' # low pt pho 2-15 GeV
    data_list_pho = openFiles(inpath_pho, desc='Loading photon files')

    ## input files pions
    inpath_pi = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/ntup_pi_18062024/'
    data_list_pi = openFiles(inpath_pi, desc='Loading pions files')


    ## plots
    print('doing plots...')

    ### --- standard plots

    print('doing histos...')
    doHisto(data_list_pho, data_list_pi, out_dir, norm=False)
    doHisto(data_list_pho, data_list_pi, out_dir, norm=True)

    print('doing ENprofile...')
    doENprofile(data_list_pho, data_list_pi, out_dir)

    print('doing Gun plots...')
    doGunPlots(data_list_pho, data_list_pi, out_dir)

    print('doing mult plot...')
    doMultiplicityPlots(data_list_pho, data_list_pi, out_dir)

    print('doing mult plot per category...')
    doMultiplicityPlots_cat(data_list_pho, data_list_pi, out_dir)

    print('doing visualization plots...')
    doVisualizationPlots(data_list_pho, data_list_pi, out_dir)

    print('plotting fraction in CEH ...')
    plotFractionCEH(data_list_pho, data_list_pi, out_dir)


    ### --- checks for special events

    # ## CAUTION: it produces a loooot of plots
    # print('checking shower extension...')
    # checkShowerExt(data_list_pho, out_dir)

    # ## CAUTION: it produces a loooot of plots
    # print('checking fraction CEH for photons...')
    # checkFractionCEH(data_list_pho, out_dir, 'orange')
    # print('checking fraction CEH for pions...')
    # checkFractionCEH(data_list_pi, out_dir, 'green')


    ### --- checks for training samples

    # print('checking early showering pions')
    # plot_showerExtCEE(data_list_pi, out_dir)

    # print('checking pions with enough energy in CEE')
    # plot_enFracCEE(data_list_pi, out_dir)

    # print('checking pions with enough energy in CEE and shower extension')
    # plot_traincheck_all(data_list_pi, out_dir)
