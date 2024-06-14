#!/usr/bin/env/python3

import glob
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import torch
#import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

plt.style.use(hep.style.CMS)
mpl.use('agg')



def openFiles(path: str, desc: str = 'Opening files') -> List[Data]:
    """
    Function to open files

    Arguments
    ---------
    path : str
        Path to the input files
    desc : str
        Description for the tqdm progress bar

    Returns
    -------
    data_list : List[Data]
        List of Data objects
    """
    data_list = []
    filename_list = [f for f in glob.glob(path + 'data_*.pt')]
    filename_list = tqdm(filename_list, desc=desc, unit='file(s)')
    for i in filename_list:
        idx = torch.load(i)
        data_list.append(idx)
    return data_list
