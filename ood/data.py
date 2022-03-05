from os import path
from torch import nn
from torch.utils.data import Dataset
import json
import os
import numpy as np

import sys


class EmbDataset(Dataset):
    def __init__(self, path_to_embs, emb_size):
        self.emb_size = emb_size
        self.embs = np.load(path_to_embs)

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        return self.embs[idx, : self.emb_size], int(self.embs[idx, self.emb_size])

class EmbDatasetOOD(Dataset):
    def __init__(self, path_to_in_distr_embs, path_to_out_distr_embs, emb_size):
        self.emb_size = emb_size

        # 0 class for in distr samples 
        in_distr_embs = np.load(path_to_in_distr_embs)
        in_distr_embs[:,self.emb_size] = 0  

        out_distr_embs = np.load(path_to_out_distr_embs)
        out_distr_embs[:,self.emb_size] = 1  
        
        self.embs = np.concatenate([in_distr_embs, out_distr_embs], axis=0)

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        return self.embs[idx, : self.emb_size], int(self.embs[idx, self.emb_size])