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
        return self.embs[idx,:self.emb_size], int(self.embs[idx,self.emb_size]) 
