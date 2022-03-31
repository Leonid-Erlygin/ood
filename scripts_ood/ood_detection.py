import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest


import sys

sys.path.append("/workspaces/ood/")

from ood.eval import accuracy
from ood.models import LinearClass
from ood.data import EmbDataset
from ood.eval import evaluate_linearmodel, compute_energy


def isolation_forest_scores(
    in_distr_train_path,
    in_distr_test_path,
    out_distr_test_path,
    n_estimators,
    max_samples,
    max_features,
    in_distr_ds_name,
    out_distr_ds_name,
    verbose,
    seed,
):
    # load emb without label
    in_distr_train = np.load(in_distr_train_path)
    in_distr_train = in_distr_train[:, :-1]

    in_distr_test = np.load(in_distr_test_path)
    in_distr_test = in_distr_test[:, :-1]

    out_distr_test = np.load(out_distr_test_path)
    out_distr_test = out_distr_test[:, :-1]

    rng = np.random.RandomState(seed)

    clf = IsolationForest(
        n_estimators,
        max_samples=max_samples,
        random_state=rng,
        verbose=verbose,
        max_features=max_features,
        n_jobs=-1,
    )
    clf.fit(in_distr_train)

    in_distr_test_scores = clf.score_samples(in_distr_test)
    out_distr_test_scores = clf.score_samples(out_distr_test)

    return {
        in_distr_ds_name: -in_distr_test_scores,
        out_distr_ds_name: -out_distr_test_scores,
    }


def compute_softmax_and_energy(
    emb_size,
    num_classes,
    linear_model_path,
    in_distr_test_path,
    out_distr_test_path,
    device,
    in_distr_ds_name,
    out_distr_ds_name,
):
    model = LinearClass(emb_size=emb_size, num_classes=num_classes)
    model.load_state_dict(torch.load(linear_model_path))
    model.eval().to(device)

    in_distr_test = EmbDataset(in_distr_test_path, emb_size=emb_size)
    out_distr_test = EmbDataset(out_distr_test_path, emb_size=emb_size)

    in_distr_test_dataloader = DataLoader(in_distr_test, batch_size=256, shuffle=False)
    accuracy_test = accuracy(model, in_distr_test_dataloader, device)

    test_accuracy = str(np.round(accuracy_test, 5)).ljust(7, "0")
    print(f"Linear model in distribution test accuracy: {test_accuracy}", flush=True)
    print("Computing softmax and energy scores...", flush=True)

    max_softmax_values_cifar, argmax_softmax_values_cifar = evaluate_linearmodel(
        model, in_distr_test, device
    )
    max_softmax_values_svhn, argmax_softmax_values_svhn = evaluate_linearmodel(
        model, out_distr_test, device
    )

    softmax_scores_distr = {
        in_distr_ds_name: max_softmax_values_cifar,
        out_distr_ds_name: max_softmax_values_svhn,
    }

    energy_values_cifar = compute_energy(model, in_distr_test, device)
    energy_values_svhn = compute_energy(model, out_distr_test, device)

    energy_distr = {
        in_distr_ds_name: -energy_values_cifar,
        out_distr_ds_name: -energy_values_svhn,
    }

    return softmax_scores_distr, energy_distr
