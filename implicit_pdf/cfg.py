"""Sample configuration dataclass for training repo."""
from typing import List, Optional, Literal, Union
from dataclasses import dataclass, field
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from pathlib import Path
import pyrallis
from enum import Enum
from implicit_pdf.utils import l2, zero, accuracy

class Wrap:
    """wrapper for serializing/deserializing classes"""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return repr(self.fn)


class LRMethod(Enum):
    """Enum class for lr methods, used with Wrap"""

    onecycle: Wrap = Wrap(OneCycleLR)
    constant: Wrap = Wrap(LambdaLR)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class Criterion(Enum):
    """Enum class for criterion, used with Wrap"""

    mse = Wrap(torch.nn.functional.mse_loss)
    l1 = Wrap(torch.nn.functional.l1_loss)
    l2 = Wrap(l2)
    zero = Wrap(zero)
    crossentropy = Wrap(torch.nn.CrossEntropyLoss())
    accuracy = Wrap(accuracy)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


@dataclass()
class DataConfig:
    """config for model specification"""

    # root dir of train dataset
    train_root: Union[str, Path] = "~/data/datasets/symsol_1.0.0/train"
    # root dir of test dataset
    test_root: Optional[Union[Path, str]] = "~/data/datasets/symsol_1.0.0/test"
    # symmetric solid object string (tet, cube, icosa, cone, cyl, sphere)
    symsol_object: str = "tet"
    # shuffle dataloaders
    shuffle: bool = True
    # use subset of dataset if length > -1
    length: int = -1


@dataclass()
class LogConfig:
    """config for logging specification"""

    # mlflow tracking uri
    uri: Optional[str] = "~/dev/implicit-pdf/implicit_pdf/mlruns"
    # toggle asynchronous logging (not implemented in dl_schema)
    enable_async: bool = True
    # every `train_freq` steps, log training quantities (metrics, single image batch, etc.)
    train_freq: int = 100
    # every `test_freq` steps, log test quantities (metrics, single image batch, pdf plots, etc.)
    test_freq: int = 500
    # save plot of p(R|x) for each image x in batch
    plot_pdf: bool = True
    # number of images to include in pdf grid plot
    n_pdf_samples: int = 6
    # every `save_freq` steps save model checkpoint according to save criteria
    save_freq: int = 1000
    # save initial model state
    save_init: bool = False
    # save last model state
    save_last: bool = False
    # save best model state (early stopping)
    save_best: bool = False


@dataclass()
class TrainConfig:
    """config for training instance"""

    # config for data specification
    data: DataConfig = field(default_factory=DataConfig)
    # config for logging specification
    log: LogConfig = field(default_factory=LogConfig)
    # run name
    run_name: str = "run_0"
    # experiment name
    exp_name: str = "debug"
    # gpu list to expose to training instance
    gpus: List[int] = field(default_factory=lambda: [-1])
    # random seed, set to make deterministic
    seed: int = 42
    # number of cpu workers in dataloader
    num_workers: int = 4
    # number of training steps (weight updates)
    train_steps: int = 1000
    # batch size
    bs: int = 32
    # number of rotation queries during training (populates SO3 and provides normalization)
    num_train_queries: int = 2 ** 12
    # number of rotation queries during evaluation (populates SO3 and provides normalization)
    num_eval_queries: int = 2 ** 16
    # length of image feature vector (512 or 2048 for resnets)
    len_img_feature: int = 512
    # rotation representation dimension
    rot_dims: int = 9
    # sizes of fully connected layers in SO3MLP
    fc_sizes: List[int] = field(default_factory=lambda: [256, 256])
    # learning rate (if onecycle, max_lr)
    lr: float = 3e-4
    # lr schedule type: (constant, onecycle)
    lr_method: LRMethod = LRMethod.onecycle
    # initial lr = lr / div
    onecycle_div_factor: float = 25
    # final lr = lr / final_div
    onecycle_final_div_factor: float = 1e4
    # weight decay as used in AdamW
    weight_decay: float = 0.0  # only applied on matmul weights
    # adamw momentum parameters
    betas: tuple = (0.9, 0.95)
    # checkpoint load path
    load_ckpt_pth: Optional[str] = None
    # load optimizer along with weights
    load_optimizer: bool = False
    # resume from last saved epoch in ckpt
    resume: bool = False
    # metric function 1: (l1, l2, mse, zero)
    metric1: Criterion = Criterion.zero
    # activate torch anomaly detection for debugging
    anomaly_detection: bool = False


if __name__ == "__main__":
    """test the train config, export to yaml"""
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("train_cfg.yaml", "w"))
