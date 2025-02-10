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

from plotter_functions_4samples import *
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
    inpath_pho    = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/S2Rmin_pho_30072024/' # normal photons
    data_list_pho = openFiles(inpath_pho, desc='Loading normal photon files')

    ## input files low energy photons
    inpath_pho_lowEn    = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/S2Rmin_pho2to15_27092024/' # low pt pho 2-15 GeV
    data_list_pho_lowEn = openFiles(inpath_pho_lowEn, desc='Loading low energy photon files')

    ## input files pions w/o high energy early showering pions
    inpath_pi_hizpu    = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/S2R0p9_pi_30072024/'
    data_list_pi_hizpu = openFiles(inpath_pi_hizpu, desc='Loading pions hizpu files')

    ## input files pions low energy clusters
    inpath_pi_lowzpu    = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/S2Rno_PUenrich_05082024/'
    data_list_pi_lowzpu = openFiles(inpath_pi_lowzpu, desc='Loading pions lowzpu files')


    ## list of data lists
    data_list   = [data_list_pho,
                   data_list_pho_lowEn,
                   data_list_pi_hizpu,
                   data_list_pi_lowzpu]
    labels_list = ['pho', 'pholowen', 'pihizpu', 'pilowzpu']
    ## dict of data
    data_dict = dict(zip(labels_list, data_list))


    ## plots
    print('doing plots...')

    ### --- standard plots

    print('doing histos...')
    doHisto(data_dict, out_dir, norm=True)

    print('doing Gun plots...')
    doGunPlots(data_dict, out_dir, norm=True)

    print('plot barycenter...')
    doBarycenter(data_dict, out_dir, norm=True)
