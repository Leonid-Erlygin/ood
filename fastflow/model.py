"""FASTFLOW: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

[FASTFLOW-AD](https://arxiv.org/pdf/2111.07677.pdf)
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Tuple, Union, cast

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, nn, optim

from anomalib.core.model import AnomalyModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.models.fastflow.backbone import fastflow_head

__all__ = ["AnomalyMapGenerator", "FastflowModel", "FastflowLightning"]


def get_logp(
    dim_feature_vector: int, p_u: torch.Tensor, logdet_j: torch.Tensor
) -> torch.Tensor:
    """Returns the log likelihood estimation.

    Args:
        dim_feature_vector (int): Dimensions of the condition vector
        p_u (torch.Tensor): Random variable u
        logdet_j (torch.Tensor): log of determinant of jacobian returned from the invertable decoder

    Returns:
        torch.Tensor: Log probability
    """
    logp = (
        torch.mean(0.5 * torch.sum(p_u**2, dim=(1, 2, 3)) - logdet_j) / p_u.shape[1]
    )
    return logp


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
        pool_layers: List[str],
    ):
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = (
            image_size if isinstance(image_size, tuple) else tuple(image_size)
        )
        self.pool_layers: List[str] = pool_layers

    def compute_anomaly_map(
        self,
        distribution: Union[List[Tensor], List[List]],
        height: List[int],
        width: List[int],
    ) -> Tensor:
        """Compute the layer map based on likelihood estimation.

        Args:
          distribution: Probability distribution for each decoder block
          height: blocks height
          width: blocks width

        Returns:
          Final Anomaly Map

        """

        test_map: List[Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            test_norm = torch.tensor(
                distribution[layer_idx], dtype=torch.double
            )  # pylint: disable=not-callable
            test_norm -= torch.max(
                test_norm
            )  # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_mask = torch.exp(test_norm)  # convert to probs in range [0:1]
            # upsample
            test_map.append(
                F.interpolate(
                    test_mask.unsqueeze(1),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()
            )
        # score aggregation
        score_map = torch.zeros_like(test_map[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += test_map[layer_idx]
        score_mask = score_map
        # invert probs to anomaly scores
        anomaly_map = score_mask.max() - score_mask

        return anomaly_map

    def __call__(self, **kwargs: Union[List[Tensor], List[int], List[List]]) -> Tensor:
        """Returns anomaly_map.

        Expects `distribution`, `height` and 'width' keywords to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size),
        >>>        pool_layers=pool_layers)
        >>> output = self.anomaly_map_generator(distribution=dist, height=height, width=width)

        Raises:
            ValueError: `distribution`, `height` and 'width' keys are not found

        Returns:
            torch.Tensor: anomaly map
        """
        if not ("distribution" in kwargs and "height" in kwargs and "width" in kwargs):
            raise KeyError(
                f"Expected keys `distribution`, `height` and `width`. Found {kwargs.keys()}"
            )

        # placate mypy
        distribution: List[Tensor] = cast(List[Tensor], kwargs["distribution"])
        height: List[int] = cast(List[int], kwargs["height"])
        width: List[int] = cast(List[int], kwargs["width"])
        return self.compute_anomaly_map(distribution, height, width)


class FastflowModel(nn.Module):
    """FASTFLOW"""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        dims = [32, 16, 8]

        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.fiber_batch_size = hparams.dataset.fiber_batch_size
        self.condition_vector: int = hparams.model.condition_vector
        self.dec_arch = hparams.model.decoder
        self.pool_layers = hparams.model.layers

        self.encoder = FeatureExtractor(
            backbone=self.backbone(pretrained=True), layers=self.pool_layers
        )
        self.pool_dims = self.encoder.out_dims
        self.decoders = nn.ModuleList(
            [
                fastflow_head(
                    self.condition_vector,
                    hparams.model.coupling_blocks,
                    hparams.model.clamp_alpha,
                    pool_dim,
                    dim,
                )
                for pool_dim, dim in zip(self.pool_dims, dims)
            ]
        )

        # encoder model is fixed
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=tuple(hparams.model.input_size), pool_layers=self.pool_layers
        )

    def forward(self, images):
        """Forward-pass images into the network to extract encoder features and compute probability.

        Args:
          images: Batch of images.

        Returns:
          Predicted anomaly maps.

        """

        activation = self.encoder(images)

        distribution = []

        height: List[int] = []
        width: List[int] = []
        for layer_idx, layer in enumerate(self.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW

            (
                batch_size,
                dim_feature_vector,
                im_height,
                im_width,
            ) = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = (
                batch_size * image_size
            )  # number of rows in the conditional vector

            height.append(im_height)
            width.append(im_width)
            decoder = self.decoders[layer_idx].to(images.device)
            # decoder returns the transformed variable z and the log Jacobian determinant
            p_u, log_jac_det = decoder(encoder_activations)

            #
            decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
            distribution.append(torch.mean(p_u, 1).detach())

        output = self.anomaly_map_generator(
            distribution=distribution, height=height, width=width
        )
        return output.to(images.device)


