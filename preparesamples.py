"""
function to prepare datasets for train and test
"""

from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm as tqdm

from tools.openFiles import openFiles


class PhoPiDataset(Dataset):

    def __init__(self, pho_dir: str, pi_dir: str):
        self.pho_dir = pho_dir
        self.pi_dir = pi_dir

        # Placeholder lists
        self.prepared = False
        self.data = None
        self.labels = None


    def _make_training_datasets_pi(self, data_list: List[Data]) -> List[np.ndarray]:
        """
        function to make pions more photon-like (applying some selection)
        """

        # loop over files in data list
        survived_evt_list = []
        for i_file in data_list:
            # loop over all events in one file
            for i_evt in i_file:

                # --- read 2D objects
                # properties: clusX,clusY,clusZ,clusE,clusT,clusL
                LCmatrix = i_evt.clus2d_feat.numpy()

                # --- select only events with shower length in CEE > 10
                # consider CEE limit: layer 27 is the first layer of CEH
                # if layer numbering starts from 1

                # here define a mask:
                # - LCmatrix[:,5] < 27: select only LCs in CEE
                # - np.unique(LCmatrix[LCmatrix[:,5] < 27][:,5]): select only unique layers (since there could be more than 1 LC in a layer)
                # - len(np.unique(LCmatrix[LCmatrix[:,5] < 27][:,5])): count the number of unique layers
                nlayers = len(np.unique(LCmatrix[LCmatrix[:,5] < 27][:,5]))

                # then require to have at least 10 layers in CEE
                if nlayers < 10:
                    continue

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

                # Whatever gets here has survived the selection
                survived_evt_list.append(LCmatrix)

        return survived_evt_list


    def prepare_data(self) -> 'PhoPiDataset':
        # Select photons
        pho_data = openFiles(self.pho_dir)
        pho_data = [i_evt.clus2d_feat.numpy() for i_file in pho_data for i_evt in i_file] # list of only LC tensors (same as pions)
        pho_labels = [1] * len(pho_data)

        # Select pions
        pi_data = openFiles(self.pi_dir)
        pi_data = self._make_training_datasets_pi(pi_data)
        pi_labels = [0] * len(pi_data)

        # Save the data
        self.data = pho_data + pi_data
        self.labels = pho_labels + pi_labels
        self.prepared = True

        # Save max shape
        max_shape = max([i.shape[0] for i in self.data])

        # Pad each entry to the max shape
        self.data = [np.pad(i, ((0, max_shape - i.shape[0]), (0, 0))) for i in self.data]

        # To select the data later...
        # pho = self.data[self.labels == 1]
        # pi = self.data[self.labels == 0]

        return self


    def __len__(self) -> int:
        if not self.prepared:
            raise ValueError("Data has not been prepared! Call the prepare_data method first!")
        return len(self.data)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if not self.prepared:
            raise ValueError("Data has not been prepared! Call the prepare_data method first!")
        return torch.from_numpy(self.data[idx]), self.labels[idx]


if __name__ == '__main__':
    pho_dir = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/ntup_pho_18082024/'
    pi_dir = '/grid_mnt/data__data.polcms/cms/sghosh/NEWPID_TICLDUMPER_DATA/ntup_pi_18062024/'

    # test
    dataset = PhoPiDataset(pho_dir, pi_dir).prepare_data()
    print('Length of dataset:', len(dataset))
    print('First item shape:', dataset[0][0].shape)

    # # Create the dataset
    # train_dataset = PhoPiDataset(pho_train_dir, pi_train_dir).prepare_data()
    # val_dataset = PhoPiDataset(pho_val_dir, pi_val_dir).prepare_data()

#    # Get the data
#    data = train_dataset[0]
#    print(data[0].shape, data[1])
#
#    # Loop over the data
#    for i in data:
#        print(i[0].shape, i[1])
