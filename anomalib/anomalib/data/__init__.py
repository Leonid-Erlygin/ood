"""Anomalib Datasets."""

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

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .mvtec import MVTecDataModule
from .precompute_emb import WideResnet_emb


def get_datamodule(config: Union[DictConfig, ListConfig]) -> LightningDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    datamodule: LightningDataModule

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTecDataModule(
            # TODO: Remove config values. IAAALD-211
            root=config.dataset.path,
            category=config.dataset.category,
            task=config.dataset.task,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[0]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.project.seed,
        )
    elif config.dataset.format.lower() == "precompute_wide_resnet50_2":
        datamodule = WideResnet_emb(
            train_dataset_path=config.dataset.train_dataset_path,
            test_dataset_in_path=config.dataset.test_dataset_in_path,
            test_dataset_out_path=config.dataset.test_dataset_out_path,
            layers=config.model.layers,
            shapes=config.model.pool_dims,
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            train_size=config.dataset.train_size,
            test_size=config.dataset.test_size,
            num_workers=config.dataset.num_workers,
            create_validation_set=config.dataset.create_validation_set,
        )
    else:
        raise ValueError("Unknown dataset!")

    return datamodule
