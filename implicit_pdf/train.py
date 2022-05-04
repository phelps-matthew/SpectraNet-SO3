"""Sample training run"""
import logging
import os
from pathlib import Path

import mlflow
import pyrallis
from torchvision import models
import torch

from implicit_pdf.cfg import TrainConfig
from implicit_pdf.dataset import SymSolDataset
from implicit_pdf.models.so3mlp import SO3MLP
from implicit_pdf.trainer import Trainer
from implicit_pdf.utils import flatten, set_seed
from implicit_pdf.recorder import Recorder

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    cfg = pyrallis.parse(config_class=TrainConfig)

    # make deterministic
    set_seed(cfg.seed)

    # set GPU
    gpus = ",".join([str(i) for i in cfg.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    logger.info(f"setting gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # create datasets
    logger.info("loading datasets")
    train_dataset, test_dataset = None, None
    if (
        cfg.data.train_root is not None
        and Path(cfg.data.train_root).expanduser().exists()
    ):
        train_dataset = SymSolDataset(split="train", cfg=cfg)
    if (
        cfg.data.test_root is not None
        and Path(cfg.data.test_root).expanduser().exists()
    ):
        test_dataset = SymSolDataset(split="test", cfg=cfg)

    # create recorder and start mlflow run
    recorder = Recorder(cfg)
    recorder.create_experiment()
    with recorder.start_run():
        # activate debug mode if necessary
        if cfg.anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

        # build vision model
        logger.info("initializing ResNet18 image model")
        img_model = models.resnet18(pretrained=True)
        img_model.fc = torch.nn.Identity()

        # build implicit pdf model
        logger.info("initializing SO3MLP implicit pdf model")
        implicit_model = SO3MLP(cfg)

        # initialize Trainer
        logger.info("initializing trainer")
        if train_dataset is None and test_dataset is None:
            logger.error("no datasets passed!")
        trainer = Trainer(cfg, img_model, implicit_model, train_dataset, test_dataset, recorder)

        # log params, state dicts, and relevant training scripts to mlflow
        script_dir = Path(__file__).parent
        cfg_dict = pyrallis.encode(cfg)  # cfg as dict, encoded for yaml
        recorder.log_artifact(script_dir / "cfg.py", "archive")
        recorder.log_artifact(script_dir / "dataset.py", "archive")
        recorder.log_artifact(script_dir / "recorder.py", "archive")
        recorder.log_artifact(script_dir / "train.py", "archive")
        recorder.log_artifact(script_dir / "trainer.py", "archive")
        recorder.log_artifact(script_dir / "utils.py", "archive")
        recorder.log_artifact(script_dir / "models/so3mlp.py", "archive")
        recorder.log_artifact(script_dir / "models/so3pdf.py", "archive")
        recorder.log_dict(cfg_dict, "archive/cfg.yaml")
        recorder.log_params(flatten(cfg_dict))

        # train
        if cfg.load_ckpt_pth:
            trainer.load_model()
        if cfg.log.save_init:
            trainer.save_model("init.pt")
        trainer.run()

        # stop mlflow run
        recorder.end_run()


if __name__ == "__main__":
    main()
