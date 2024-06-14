#!/usr/bin/env/python3

## ---
#  script to plot variables from training samples
#  run with: python3 plotter_pho-pi_samples.py
## ---

import os
from datetime import date
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

# import modules
from plotter_functions import *

plt.style.use(hep.style.CMS)
mpl.use('agg')





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

    #print('doing histos...')
    #doHisto(data_list_pho, data_list_pi, out_dir, False)

    #print('doing ENprofile...')
    #doENprofile(data_list_pho, data_list_pi, out_dir)

    #print('doing Gun plots...')
    #doGunPlots(data_list_pho, data_list_pi, out_dir)

    print('doing mult plot...')
    doMultiplicityPlots(data_list_pho, data_list_pi, out_dir)

    print('doing mult plot per category...')
    doMultiplicityPlots_cat(data_list_pho, data_list_pi, out_dir)

    print('doing visualization plots...')
    doVisualizationPlots(data_list_pho, data_list_pi, out_dir)
