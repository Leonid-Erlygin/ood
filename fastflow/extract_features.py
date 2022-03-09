import torch
from tqdm import tqdm
import numpy as np
import pickle
from feature_extractor import FeatureExtractor


def create_feature_dataset(model, layers, dataset, out_name, num_images_per_class, device):
    encoder = FeatureExtractor(model, layers)
    encoder.eval()
    
    print(f'saving to {out_name}...')
    preds = {layer : [] for layer in layers}

    label_counts = {i:0 for i in range(10)}
    for image, label in tqdm(dataset):
        if label_counts[label] > num_images_per_class:
            continue
        features = encoder(torch.unsqueeze(image.to(device), dim=0))
        for layer in features.keys():
            preds[layer].append(features[layer].detach().cpu())
        label_counts[label]+=1
    
    for layer in preds.keys():
        preds[layer] = torch.cat(preds[layer]).numpy()

    np.savez_compressed(out_name, **preds)