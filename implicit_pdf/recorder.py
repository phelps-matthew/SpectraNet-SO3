"""Logging utility for training runs"""
import math
import logging
import json
from pathlib import Path
import numpy as np
import mlflow
import torchvision
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from implicit_pdf.recorder_base import RecorderBase, AsyncCaller

logger = logging.getLogger(__name__)


class Recorder(RecorderBase):
    """artifact logger for spec"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_iter(self, inputs, step=None, batch=True, split="train"):
        """log batch artifacts
        Args:
            inputs: dictionary of torch inputs from run_epoch
            step: iteration step
            batch: batch iteration or epoch iteration
            split: train, test, or infer split
        """
        pass

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_image_grid(
        self,
        x,
        prefix="train",
        suffix="",
    ):
        """log batch of images"""
        n_rows = math.ceil(math.sqrt(self.cfg.bs))  # actually n_cols

        # log images. these are (N, C, H, W) torch.float32
        if x is not None:
            grid_x = torchvision.utils.make_grid(
                x, normalize=True, nrow=n_rows, pad_value=1.0, padding=2
            ).permute(1, 2, 0)
            self.client.log_image(
                self.run_id, grid_x.numpy(), f"{prefix}_x_{suffix}.jpg"
            )
