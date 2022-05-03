"""Logging utility for training runs"""
import math
import logging
from pathlib import Path
import numpy as np
import torchvision
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from implicit_pdf.recorder_base import RecorderBase, AsyncCaller
from implicit_pdf.utils import euler_to_so3, so3_to_euler
import io

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

    def figure_to_array(self, figure):
        """convert matplotlib figure to numpy array"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close(figure)
        buffer.seek(0)
        # save matplotlib figure to np.ndarray
        data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        # reshape from (n) to (3, H, W)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def plot_pdf_panel(
        self, images, probabilities, rotations, query_rotations, n_samples=5
    ):
        """
        Plot panel of raw image and so3 distribution, one panel per image.

        Args:
            images: (N, C, H, W)
            probabilities: (N, 1)
            rotations: ground truth rotations, (N, 3, 3)
            query_rotations: rotations used to construct pdf, (N, n_queries, 3, 3)
            n_samples: plot n_samples pdfs viz. images[:n_samples]
        Returns:
            figures: list of matplotlib figures
        """
        figure_list = []
        inches_per_subplot = 4
        canonical_rotation = np.float32(euler_to_so3(np.array([0.2] * 3)))
        canonical_rotation = torch.from_numpy(canonical_rotation)
        for img_idx in range(n_samples):
            fig = plt.figure(
                figsize=(3 * inches_per_subplot, inches_per_subplot), dpi=100
            )
            gs = fig.add_gridspec(1, 3)
            fig.add_subplot(gs[0, 0])
            plt.imshow(images[img_idx].permute(1, 2, 0))
            plt.axis("off")
            ax2 = fig.add_subplot(gs[0, 1:], projection="mollweide")
            figure_i = self.plot_pdf(
                query_rotations,
                probabilities[img_idx],
                rotations[img_idx],
                ax=ax2,
                fig=fig,
                display_threshold_probability=1e-2 / query_rotations.shape[0],
                canonical_rotation=canonical_rotation,
            )
            figure_list.append(figure_i)
        return torch.cat(figure_list, 0)

    def plot_pdf(
        self,
        rotations,
        probabilities,
        rotations_gt=None,
        ax=None,
        fig=None,
        display_threshold_probability=0,
        to_image=True,
        show_color_wheel=True,
        canonical_rotation=torch.eye(3),
    ):
        """Plot a single distribution on SO(3) using the tilt-colored method.

        Args:
          rotations: [N, 3, 3] tensor of rotation matrices
          probabilities: [N] tensor of probabilities
          rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
          ax: The matplotlib.pyplot.axis object to paint
          fig: The matplotlib.pyplot.figure object to paint
          display_threshold_probability: The probability threshold below which to omit
            the marker
          to_image: If True, return a tensor containing the pixels of the finished
            figure; if False return the figure itself
          show_color_wheel: If True, display the explanatory color wheel which matches
            color on the plot with tilt angle
          canonical_rotation: A [3, 3] rotation matrix representing the 'display
            rotation', to change the view of the distribution.  It rotates the
            canonical axes so that the view of SO(3) on the plot is different, which
            can help obtain a more informative view.

        Returns:
          A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
        """

        def _show_single_marker(
            ax, rotation, marker, edgecolors=True, facecolors=False
        ):
            eulers = so3_to_euler(rotation)
            xyz = rotation[:, 0]
            tilt_angle = eulers[0].item()
            longitude = np.arctan2(xyz[0], -xyz[1])
            latitude = np.arcsin(xyz[2])

            color = cmap(0.5 + tilt_angle / 2 / np.pi)
            ax.scatter(
                longitude,
                latitude,
                s=2500,
                edgecolors=color if edgecolors else "none",
                facecolors=facecolors if facecolors else "none",
                marker=marker,
                linewidth=4,
            )

        if ax is None:
            fig = plt.figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111, projection="mollweide")
        if rotations_gt is not None and rotations_gt.dim() == 2:
            rotations_gt = rotations_gt[None, :]

        # (n_queries, 3, 3)
        display_rotations = torch.matmul(rotations, canonical_rotation)
        cmap = plt.cm.hsv
        scatterpoint_scaling = 4e3
        # (n_queries, 3)
        eulers_queries = so3_to_euler(display_rotations)
        # first column of 3x3 rot matrix (n_queries, 3)
        xyz = display_rotations[:, :, 0]
        # roll, or angle corresponding to R_x
        tilt_angles = eulers_queries[:, 0]

        longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
        latitudes = np.arcsin(xyz[:, 2])

        which_to_display = probabilities > display_threshold_probability

        if rotations_gt is not None:
            # The visualization is more comprehensible if the GT
            # rotation markers are behind the output with white filling the interior.
            # (N, 3, 3)
            display_rotations_gt = torch.matmul(rotations_gt, canonical_rotation)

            for rotation in display_rotations_gt:
                _show_single_marker(ax, rotation, "o")
            # Cover up the centers with white markers
            for rotation in display_rotations_gt:
                _show_single_marker(
                    ax, rotation, "o", edgecolors=False, facecolors="#ffffff"
                )

        # Display the distribution
        # fmt: off
        import ipdb; ipdb.set_trace(context=30)  # noqa
        # fmt: on
        ax.scatter(
            longitudes[which_to_display],
            latitudes[which_to_display],
            s=scatterpoint_scaling * probabilities[which_to_display],
            c=cmap(0.5 + tilt_angles[which_to_display] / 2.0 / np.pi),
        )

        ax.grid()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if show_color_wheel:
            # Add a color wheel showing the tilt angle to color conversion.
            ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
            theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
            radii = np.linspace(0.4, 0.5, 2)
            _, theta_grid = np.meshgrid(radii, theta)
            colormap_val = 0.5 + theta_grid / np.pi / 2.0
            ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
            ax.set_yticklabels([])
            ax.set_xticklabels(
                [
                    r"90$\degree$",
                    None,
                    r"180$\degree$",
                    None,
                    r"270$\degree$",
                    None,
                    r"0$\degree$",
                ],
                fontsize=14,
            )
            ax.spines["polar"].set_visible(False)
            plt.text(
                0.5,
                0.5,
                "Tilt",
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        if to_image:
            return self.figure_to_array(fig)
        else:
            return fig
