import numpy as np
from torch.optim.lr_scheduler import StepLR
from scripts_ood.models import LinearClass
from torch import nn
from torch.utils.data import DataLoader
import torch

from scripts_ood.data import EmbDataset
from scripts_ood.eval import accuracy


def init_linear_train(
    in_distr_train_path,
    in_distr_val_path,
    emb_size,
    num_classes,
    init_lr,
    lr_step_size,
    lr_gamma,
    weight_decay,
    device,
    batch_size,
):
    cifar_train = EmbDataset(in_distr_train_path, emb_size=emb_size)
    cifar_test = EmbDataset(in_distr_val_path, emb_size=emb_size)

    soft_cifar_linear_model = LinearClass(
        emb_size=emb_size, num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        soft_cifar_linear_model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    cifar_train_dataloader = DataLoader(
        cifar_train, batch_size=batch_size, shuffle=True
    )
    cifar_val_dataloader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    return (
        soft_cifar_linear_model,
        cifar_train_dataloader,
        cifar_val_dataloader,
        optimizer,
        criterion,
        scheduler,
    )


def train_linear(
    train_dataloader,
    test_dataloader,
    model,
    optimizer,
    scheduler,
    criterion,
    metric,
    train_epoch,
    metric_freq,
    device,
):
    model.train()

    for epoch in range(train_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss += loss.item()
            if i % metric_freq == 0:
                accuracy_train = metric(model, train_dataloader, device)
                accuracy_test = metric(model, test_dataloader, device)
                train_loss = str(np.round(running_loss / (i + 1), 5)).ljust(7, "0")
                train_accuracy = str(np.round(accuracy_train, 5)).ljust(7, "0")
                test_accuracy = str(np.round(accuracy_test, 5)).ljust(7, "0")

                print(
                    f"epoch: {epoch}, train loss: {train_loss},  train_accuracy: {train_accuracy}, test_accuracy {test_accuracy}"
                )
        scheduler.step()
        print(f"current LR: {scheduler.get_lr()[0]}")


def train_linear_model(model_name, device, emb_size, in_dist_data='cifar', train_epoch=200, init_lr=1e-4):
    (
        linear_model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        scheduler,
    ) = init_linear_train(
        in_distr_train_path=f"../data/predictions/{model_name}_{in_dist_data}_train.npy",
        in_distr_val_path=f"../data/predictions/{model_name}_{in_dist_data}_test.npy",
        emb_size=emb_size,
        num_classes=10,
        init_lr=init_lr,
        lr_step_size=20,
        lr_gamma=0.5,
        weight_decay=1e-6,
        batch_size=512,
        device=device,
    )
    train_linear(
        train_dataloader,
        val_dataloader,
        linear_model,
        optimizer,
        scheduler,
        criterion,
        accuracy,
        train_epoch=train_epoch,
        metric_freq=1500,
        device=device,
    )
    return linear_model
