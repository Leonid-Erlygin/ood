from torch import nn
import torch
import torch.nn.functional as F
import torchvision
import sys

sys.path.append("/workspaces/ood/")


# from differnet.freia_funcs import (
#     permute_layer,
#     glow_coupling_layer,
#     F_fully_connected,
#     ReversibleGraphNet,
#     OutputNode,
#     InputNode,
#     Node,
# )

from configs import train_differnet as c


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


class LinearClass(nn.Module):
    """LinearClass model"""

    def __init__(self, emb_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(emb_size, num_classes, bias=True)
        self.bn = nn.BatchNorm1d(num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, batch):
        return self.bn(self.linear(batch))
