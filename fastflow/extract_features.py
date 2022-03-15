import torch
from tqdm import tqdm
import numpy as np
import pickle
from feature_extractor import FeatureExtractor


def split_layers(path_to_emb, layers):
    with np.load(path_to_emb) as data:
        for layer in tqdm(layers):
            out_name = path_to_emb.replace("_".join(layers), layer).replace(
                "npz", "npy"
            )
            np.save(out_name, data[layer])


def create_feature_dataset(
    model, layers, out_dims, dataset, out_name, num_images_per_class, device
):
    encoder = FeatureExtractor(model, layers)
    encoder.eval()

    preds = {
        layers[i]: np.zeros([num_images_per_class * 10] + out_dims[i], dtype=np.float32)
        for i in range(len(layers))
    }
    label_counts = {i: 0 for i in range(10)}
    label_not_finish = [True for _ in range(10)]

    i = 0
    for image, label in tqdm(dataset):
        if not any(label_not_finish):
            print("all classes are computed")
            break
        if label_counts[label] >= num_images_per_class:
            label_not_finish[label] = False
            continue

        features = encoder(torch.unsqueeze(image.to(device), dim=0))
        for layer in features.keys():
            feature = features[layer].detach().cpu().numpy()[0]
            print(feature.dtype)
            return
            preds[layer][i] = features[layer].detach().cpu().numpy()[0]
        i += 1
        label_counts[label] += 1
    print(f"saving to {out_name}...")
    np.savez_compressed(out_name, **preds)
