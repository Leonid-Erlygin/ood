from torch.utils.data import Dataset
import numpy as np


class FeaturesTrainDataset(Dataset):
    def __init__(self, path_to_embs, layers, shapes):
        self.layers = layers
        self.features = {}
        for layer, shape in zip(layers, shapes):
            layer_file_path = path_to_embs.replace("_".join(layers), layer)
            self.features[layer] = np.memmap(
                filename=layer_file_path, mode="r", dtype="float32", shape=shape
            )

    def __len__(self):
        return self.features[self.layers[0]].shape[0]

    def __getitem__(self, idx):
        return [np.array(self.features[layer][idx]) for layer in self.features.keys()]


class FeaturesDatasetOOD(Dataset):
    def __init__(self, path_to_in_distr_embs, path_to_out_distr_embs, layers, shapes):
        self.layers = layers

        self.features_in = {}
        self.features_out = {}
        for layer, shape in zip(layers, shapes):
            layer_file_path_in = path_to_in_distr_embs.replace("_".join(layers), layer)
            layer_file_path_out = path_to_out_distr_embs.replace(
                "_".join(layers), layer
            )
            self.features_in[layer] = np.memmap(
                filename=layer_file_path_in, mode="r", dtype="float32", shape=shape
            )
            self.features_out[layer] = np.memmap(
                filename=layer_file_path_out, mode="r", dtype="float32", shape=shape
            )

        self.num_in = self.features_in[self.layers[0]].shape[0]
        self.num_out = self.features_out[self.layers[0]].shape[0]

    def __len__(self):
        return self.num_in + self.num_out

    def __getitem__(self, idx):
        if idx < self.num_in:
            return [self.features_in[layer][idx] for layer in self.layers], 0
        else:
            return [
                self.features_out[layer][idx - self.num_in] for layer in self.layers
            ], 1
