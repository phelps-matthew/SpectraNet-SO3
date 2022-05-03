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
