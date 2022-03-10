from torch.utils.data import Dataset
import numpy as np

class FeaturesTrainDataset(Dataset):
    def __init__(self, path_to_embs, layers):
        self.layers = layers
        with np.load(path_to_embs) as data:
            self.features = {layer : data[layer] for layer in layers}
        
    def __len__(self):
        return self.features[self.layers[0]].shape[0]

    def __getitem__(self, idx):
        return [self.features[layer][idx] for layer in self.layers]


class FeaturesDatasetOOD(Dataset):
    def __init__(self, path_to_in_distr_embs, path_to_out_distr_embs, layers):
        self.layers = layers

        with np.load(path_to_in_distr_embs) as data:
            self.features_in = {layer : data[layer] for layer in layers}
        
        with np.load(path_to_out_distr_embs) as data:
            self.features_out = {layer : data[layer] for layer in layers}

        self.num_in = self.features_in[self.layers[0]].shape[0]
        self.num_out = self.features_out[self.layers[0]].shape[0]

    def __len__(self):
        return self.num_in + self.num_out

    def __getitem__(self, idx):
        if idx < self.num_in:
            return [self.features_in[layer][idx] for layer in self.layers], 0
        else:
            return [self.features_out[layer][idx - self.num_in] for layer in self.layers], 1
