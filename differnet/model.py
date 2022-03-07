import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet
import torchvision

import config as c
from freia_funcs import (
    permute_layer,
    glow_coupling_layer,
    F_fully_connected,
    ReversibleGraphNet,
    OutputNode,
    InputNode,
    Node,
)

WEIGHT_DIR = "./weights"
MODEL_DIR = "./models"


def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name="input"))
    for k in range(c.n_coupling_blocks):
        nodes.append(
            Node([nodes[-1].out0], permute_layer, {"seed": k}, name=f"permute_{k}")
        )
        nodes.append(
            Node(
                [nodes[-1].out0],
                glow_coupling_layer,
                {
                    "clamp": c.clamp_alpha,
                    "F_class": F_fully_connected,
                    "F_args": {"internal_size": c.fc_internal, "dropout": c.dropout},
                },
                name=f"fc_{k}",
            )
        )
    nodes.append(OutputNode([nodes[-1].out0], name="output"))
    coder = ReversibleGraphNet(nodes)
    return coder


class DifferNetWithEmb(nn.Module):
    def __init__(self):
        super(DifferNetWithEmb, self).__init__()

        self.linear1 = nn.Linear(c.emb_size, c.n_feat)
        self.linear2 = nn.Linear(c.n_feat, c.n_feat)
        self.sigmoid = nn.Sigmoid()
        self.nf = nf_head()

    def forward(self, x):
        # y = self.linear1(x)
        # y = self.sigmoid(y)
        # y = self.linear2(y)
        y = x
        z = self.nf(y)
        return z

class DifferNet(nn.Module):
    def __init__(self):
        super(DifferNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=False)
        model.load_state_dict(
            torch.load(
                "/workspaces/ood/data/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
            )
        )
        self.feature_extractor = model
        self.feature_extractor.to(c.device)

        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False
        self.nf = nf_head()

    def forward(self, x):
        y_cat = list()
        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2**s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))
        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z

def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model
