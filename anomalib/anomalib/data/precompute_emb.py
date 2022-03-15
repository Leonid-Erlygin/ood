import sys

sys.path.append("/workspaces/ood/")
import logging
import random
import tarfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset

from anomalib.data.transforms import PreProcessor
from anomalib.data.utils import read_image
from anomalib.utils.download_progress_bar import DownloadProgressBar

from fastflow.data import FeaturesTrainDataset, FeaturesDatasetOOD

logger = logging.getLogger(name="Dataset: MVTec")
logger.setLevel(logging.DEBUG)

__all__ = ["MVTec", "MVTecDataModule"]


class WideResnet_emb(LightningDataModule):
    """MVTec Lightning Data Module."""

    def __init__(
        self,
        train_dataset_path,
        test_dataset_in_path,
        test_dataset_out_path,
        layers,
        shapes,
        train_batch_size,
        test_batch_size,
        train_size,
        test_size,
        num_workers,
        create_validation_set,
    ) -> None:

        super().__init__()
        self.create_validation_set = create_validation_set
        self.train_dataset_path = train_dataset_path
        self.test_dataset_in_path = test_dataset_in_path
        self.test_dataset_out_path = test_dataset_out_path
        self.layers = layers
        self.shapes = shapes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)
        """
        self.test_data = FeaturesDatasetOOD(
            path_to_in_distr_embs=self.test_dataset_in_path,
            path_to_out_distr_embs=self.test_dataset_out_path,
            layers=self.layers,
            shapes=[tuple([self.test_size] + list(x)) for x in self.shapes],
        )

        self.train_data = FeaturesTrainDataset(
            path_to_embs=self.train_dataset_path,
            layers=self.layers,
            shapes=[tuple([self.train_size] + list(x)) for x in self.shapes],
        )

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    # def val_dataloader(self) -> DataLoader:
    #     """Get validation dataloader."""
    #     dataset = self.val_data if self.create_validation_set else self.test_data
    #     return DataLoader(
    #         dataset=dataset,
    #         shuffle=False,
    #         batch_size=self.test_batch_size,
    #         num_workers=self.num_workers,
    #     )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
