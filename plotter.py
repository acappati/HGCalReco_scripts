 #!/usr/bin/env/python3

## ---
#  script to plot some variables from step3 files of hgcalreco
#  run with: python3 plotter.py
## ---

import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import os
import numpy as np    
import mplhep as hep
import uproot3



plt.style.use(hep.style.CMS)



def doPlots(out_dir):

    inFile = '/data_CMS/cms/cappati/ThreePhoton_200PU/step3/step3_370.root'
    key = 'ana/hgc'

    branches = [u'vtx_x','vtx_y','vtx_z','cluster2d_eta']

    ttree = uproot3.open(inFile)[key]
    df = ttree.pandas.df(branches, entrystop=5000)
    print('Tree opened as pandas dataframe')


    # plot some float variables
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

    hep.cms.label(llabel='test', rlabel=r'$\bf{PU=?}$', ax=axs[0])
    axs[0].hist(df['vtx_x'], bins=50)
    axs[0].set_xlabel('vtx_x')
    axs[0].set_ylabel('frequency')

    hep.cms.label(llabel='test', rlabel=r'$\bf{PU=?}$', ax=axs[1])
    axs[1].hist(df['vtx_y'], bins=50)
    axs[1].set_xlabel('vtx_y')
    axs[1].set_ylabel('frequency')

    hep.cms.label(llabel='test', rlabel=r'$\bf{PU=?}$', ax=axs[2])
    axs[2].hist(df['vtx_z'], bins=50)
    axs[2].set_xlabel('vtx_z')
    axs[2].set_ylabel('frequency')

    plt.savefig(os.path.join(out_dir, 'test.png'), dpi=300)
    print('float variables plotted')


    # plot vector variables
    # 
    # better to use root. But, in case:
    # - one method is to use iloc[i] to get a certain row using the pandas row position
    # - other method is to use series and indexes
    fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

    axs1.hist(df['cluster2d_eta'].iloc[0], bins=50)

    plt.savefig(os.path.join(out_dir, 'test1.png'), dpi=300)
    print('vector variables plotted')



if __name__ == "__main__" :


    out_dir = 'plots'
    os.makedirs(out_dir,exist_ok=True) #check if output dir exist

    doPlots(out_dir)
