"""
Training executer - handles lr schedulers, optimizers, model saving/loading, 
datasets/generators, train steps, test steps, metrics, losses, etc
"""
import logging
import math
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from tqdm import tqdm

from implicit_pdf.models.so3pdf import SO3PDF
from implicit_pdf.utils import configure_adamw

logger = logging.getLogger(__name__)


class Trainer:
    """train or evaluate a dataset over n epochs"""

    def __init__(
        self,
        cfg,
        img_model,
        implicit_model,
        train_dataset,
        test_dataset=None,
        recorder=None,
        verbose=True,
    ):
        self.cfg = cfg
        self.img_model = img_model
        self.implicit_model = implicit_model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_only = self.train_dataset is None
        self.recorder = recorder
        self.verbose = verbose
        self.scheduler = None
        self.curr_step = 0

        # set mlflow paths for model/optim saving
        if recorder is not None:
            self.ckpt_root = self.recorder.root / "checkpoints"
            (self.ckpt_root).mkdir(parents=True, exist_ok=True)
        else:
            self.ckpt_root = ""

        # set gpu device if available
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.img_model = self.img_model.to(self.device)
            self.implicit_model = self.implicit_model.to(self.device)

        # initialize so3 ipdf instance
        self.so3pdf = SO3PDF(cfg, self.implicit_model, self.img_model, self.device)

        # set datloaders
        self.train_loader = self.create_dataloader(train=True)
        self.test_loader = self.create_dataloader(train=False)

        # configure optimizer
        if self.test_only:
            self.optimizer = None
            self.cfg.load_optimizer = False
        else:
            self.optimizer = configure_adamw(self.implicit_model, self.cfg)
            self.set_scheduler(steps=len(self.train_loader))

        # initialize best loss for ckpt saving
        self.best_loss = float("inf")

    def create_dataloader(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        if dataset is None:
            return None
        loader = DataLoader(
            dataset,
            shuffle=self.cfg.data.shuffle,
            pin_memory=True,
            batch_size=self.cfg.bs,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
        return loader

    def set_scheduler(self, steps):
        """create lr scheduler; steps argument required for onecycle"""
        if self.cfg.lr_method.name == "onecycle":
            self.scheduler = self.cfg.lr_method(
                self.optimizer,
                self.cfg.lr,
                total_steps=self.cfg.train_steps,
                div_factor=self.cfg.onecycle_div_factor,
                final_div_factor=self.cfg.onecycle_final_div_factor,
            )
        else:
            self.scheduler = self.cfg.lr_method(
                self.optimizer, lr_lambda=lambda epoch: 1
            )

    def save_model(self, path="last.pt", loss=None, as_artifact=True):
        """save model state dict, optim state dict, epoch and loss"""
        save_path = self.ckpt_root / path if as_artifact else path
        if self.verbose:
            logger.info(f"saving {save_path}")
        torch.save(
            {
                "model_state_dict": self.implicit_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.curr_epoch,
                "loss": loss,
            },
            save_path,
        )

    def load_model(self):
        """load model state dict, optim state dict, epoch and loss"""
        ckpt_path = Path(self.cfg.load_ckpt_pth).expanduser().resolve()
        ckpt = torch.load(ckpt_path)

        # load optimizer
        if self.cfg.load_optimizer:
            logger.info(f"loading optimizer from {ckpt_path}")
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_epoch_loss = ckpt["loss"]

        # only update scheduler and epoch counter if resuming
        if self.cfg.resume:
            logger.info(f"resuming from epoch: {ckpt['epoch']}")
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.curr_step = ckpt["step"] + 1

        # load parameters
        logger.info(f"loading model params from {ckpt_path}")
        self.implicit_model.load_state_dict(ckpt["model_state_dict"])

    def train_step(self, img_feature, y):
        self.implicit_model.train()
        with torch.enable_grad():
            probs = self.so3pdf.predict_probability(img_feature, y, train=True)
            loss = -torch.log(probs).mean()  # negative log liklihood
        # get current learning rate before optim step
        lr = self.optimizer.param_groups[0]["lr"]
        # backward step
        self.implicit_model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss, lr

    def evaluate_test_set(self):
        """evaluate over test set"""
        self.implicit_model.eval()
        losses = []
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                img_feature = self.img_model(x)
                probs = self.so3pdf.predict_probability(img_feature, y, train=False)
                loss = -torch.log(probs).mean()  # negative log liklihood
            losses.append(loss)
            pbar.set_description(f"(TEST STEP {it}: loss {loss.item():.6e}")

        # log test quantities
        mean_loss = float(np.mean(losses))
        self.recorder.log_metric("loss_test", mean_loss, self.curr_step)
        self.recorder.log_image_grid(x.detach().cpu(), name="test_x_batch")
        if self.cfg.log.plot_pdf:
            # compute pdf per image using eval rotation queries
            query_rotations, pdfs = self.so3pdf.output_pdf(img_feature)
            figures = self.recorder.plot_pdf_panel(
                images=x,
                probabilities=pdfs,
                rotations=y,
                query_rotations=query_rotations,
                n_samples=-1,
            )
            self.recorder.log_image_grid(
                torch.from_numpy(figures),
                name="test_pdf_grid",
                NCHW=False,
                normalize=False,
                jpg=False,
            )

        # model checkpointing
        if self.curr_step % self.cfg.log.save_freq == 0:
            # update best loss, possibly save best model state
            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.recorder.log_metric("loss_test-best", self.best_loss, self.curr_step)
                if self.cfg.log.save_best:
                    self.save_model("best.pt", loss=self.best_loss)
            # save latest model
            if self.cfg.log.save_last:
                self.save_model("last.pt", loss=mean_loss)

    def run(self):
        """iterate over train set and evaluate on test set"""
        # do not track gradients on image feature extractor
        self.img_model.eval()
        data_iter = iter(self.train_loader)

        # initialize running lists of quantities to be logged
        losses = []

        for step in range(self.cfg.train_steps):
            try:
                x, y = next(data_iter) 
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            self.curr_step = step
            x = x.to(self.device)
            y = y.to(self.device)

            # compute image feature vector
            with torch.no_grad():
                img_feature = self.img_model(x)

            # forward the model, calculate loss
            loss, lr = self.train_step(img_feature, y)
            losses.append(loss.item())

            # log train quantities
            if step % self.cfg.log.train_freq == 0:
                mean_train_loss = float(np.mean(losses))
                losses = []
                print(f"TRAIN STEP {step}/{self.cfg.train_steps}: lr {lr:.2e} loss {mean_train_loss:.6e}")
                self.recorder.log_metric("lr", lr, step)
                self.recorder.log_metric("loss_train", mean_train_loss, step)
                self.recorder.log_image_grid(x.detach().cpu(), name="train_x_batch")

            # evaluate test set
            if step % self.cfg.log.test_freq == 0:
                self.evaluate_test_set()
