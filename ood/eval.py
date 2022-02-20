import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('/workspaces/ood/')
from ood.ood_metrics import get_measures

def get_ood_metrics_value_range(cifar_scores, svhn_scores, fractions):
    auroces = []
    auprs = []
    fprs = []

    for frac in fractions:
        svhn_size = int(len(svhn_scores)*frac)
        auroc, aupr, fpr = get_measures(cifar_scores, svhn_scores[:svhn_size])
        auroces.append(auroc)
        auprs.append(aupr)
        fprs.append(fpr)

    return fprs, auroces, auprs

def accuracy(model, dataloader, device):
    sum = 0
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float())
        y_pred = torch.argmax(outputs, dim=1)
        sum += torch.sum(y_pred == labels).cpu().detach().numpy()
    return sum / len(dataloader.dataset)

def make_predictions(model, dataset, device):
    """
    returns logits for each samlple in dataset
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for emb, _ in tqdm(dataset):
            pred = model(torch.unsqueeze(torch.tensor(emb).float().to(device), dim=0))
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds

def evaluate_linearmodel(model, dataset, device):
    """
    returns softmax scores
    """
    preds = make_predictions(model, dataset, device)
    sm = torch.nn.Softmax(dim=1)
    preds = sm(preds)
    model_pred = np.array([x.cpu().detach().numpy() for x in preds])

    return np.max(model_pred, axis = 1), np.argmax(model_pred, axis = 1)

def compute_energy(model, dataset, device):
    """
    returns energy scores
    """
    preds = make_predictions(model, dataset, device).cpu().detach().numpy()
    preds = - np.log(np.sum(np.exp(preds), axis=-1))
    return preds
