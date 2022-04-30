"""
Class and methods to generate rotation queries, pass through SO3MLP, and yield normalized 
probability distribution function.
"""

import logging
import torch
import numpy as np
from implicit_pdf.utils import euler_to_so3

logger = logging.getLogger(__name__)


class SO3PDF:
    def __init__(self, cfg, implicit_model, img_model, device):
        self.cfg = cfg
        self.implicit_model = implicit_model
        self.img_model = img_model
        self.device = device
        self.num_train_queries = self.cfg.num_train_queries
        self.num_eval_queries = self.cfg.num_eval_queries
        # grid of equivolume SO3 queries, cached as hashmap with n_queries as keys
        self.grids = {}
        # populate grid hashmap for training and evaluation queries
        self.generate_queries(self.num_train_queries)
        self.generate_queries(self.num_eval_queries)

    def predict_probability(self, img_feature, so3, train=False):
        """Predict p(R|x), with x as img_feature. Requires global rotation transformation
        on each ground truth SO3 matrix, increasing total number of ops in first MLP layer.

        Args:
            img_feature: feature vector from image batch, (N, len_img_feature)
            so3: query rotation matrices, (N, 3, 3)
            train: defines query number based on num_train_queries or num_eval_queries

        Returns:
            p(R|x), shape (N, 1)
        """
        # generate grid of so3 queries
        if train:
            query_rotations = torch.from_numpy(
                self.generate_queries(self.num_train_queries)
            )
        else:
            query_rotations = torch.from_numpy(
                self.generate_queries(self.num_eval_queries)
            )
        num_queries = query_rotations.shape[0]
        # given grid, find requisite transformations to ensure set membership of so3 queries
        # Note: targeting last grid element [-1] is arbitrary, but helpful for bookkeeping
        delta_rot = torch.transpose(query_rotations[-1], 0, 1) @ so3
        # compute one rotated grid per batch sample
        query_rotations = torch.einsum("aij,bjk->baik", query_rotations, delta_rot)
        shape = query_rotations.shape
        # flatten rotational dimension
        query_rotations = query_rotations.reshape(shape[0], shape[1], self.cfg.rot_dims)
        # calculate unnormalized probabilities
        probabilities = self.implicit_model(
            img_feature, query_rotations, apply_softmax=True
        )
        # rescale by SO3 volume to yield normalized p(R|x)
        probabilities = probabilities * num_queries / np.pi**2
        return probabilities[:, -1]

    def predict_rotation(self, img_feature):
        """Given image feature vector, compute argmax(logits) as a point estimate of
        the predicted rotation.

        Args:
            img_feature: feature vector from image batch, (N, len_img_feature)

        Returns:
            so3_preds: SO3 rotation predictions, (N, 3, 3)
        """
        # generate queries, flatten rotation dimension
        query_rotations = torch.from_numpy(self.generate_queries(self.num_eval_queries))
        query_rotations = query_rotations.view(-1, self.cfg.rot_dims)
        logits = self.implicit_model(img_feature, query_rotations)
        # compuate argmax of logits and return highest probability rotations
        max_inds = torch.argmax(logits, dim=-1)
        max_rotations = torch.gather(query_rotations, dim=-1, index=max_inds)
        max_rotations = max_rotations.view(-1, 3, 3)
        return max_rotations

    def output_pdf(self, img_feature, num_queries=None, query_rotations=None):
        """Given image feature vector, compute normalized SO3 distribution p(R|x)

        Args:
            img_feature: feature vector from image batch, (N, len_img_feature)
            num_queries: number of queries used to evaluate pdf
            query_rotations: If given, use pre-specified rotation queries to construct pdf,
            shape (num_queries, 3, 3)

        Returns:
            probabilities (N, num_queries), query_rotations (num_queries, 3, 3)
        """
        if num_queries is None:
            num_queries = self.num_eval_queries
        if query_rotations is None:
            query_rotations = torch.from_numpy(self.generate_queries(num_queries))
        query_rotations = query_rotations.view(-1, self.cfg.rot_dims)
        probabilities = self.implicit_model(
            img_feature, query_rotations, apply_softmax=True
        )
        return query_rotations, probabilities

    def generate_queries(self, num_queries=None):
        """Generate SO3 rotation queries as equivolume grid.

        HEALPix-SO(3) is defined only on 72 * 8^N points; we find the closest valid grid size
        (in log space) to the requested size. The largest grid size tested in IPDF is 19M points.

        Returns:
            np.float32 of shape (num_queries, 3, 3)
        """
        if not num_queries:
            num_queries = self.num_eval_queries
        grid_sizes = 72 * 8 ** np.arange(7)
        size = grid_sizes[np.argmin(np.abs(np.log(num_queries) - np.log(grid_sizes)))]
        if self.grids.get(size) is not None:
            return self.grids[size]
        else:
            logging.info(f"Using grid of size {size}. Requested was {num_queries}")
            self.grids[size] = np.float32(generate_healpix_grid(size=size))
            return self.grids[size]


def generate_healpix_grid(size=None, recursion_level=None):
    """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).

    Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
    along the 'tilt' direction 6*2**recursion_level times over 2pi.

    Args:
        recursion_level: An integer which determines the level of resolution of the
        grid.  The final number of points will be 72*8**recursion_level.  A
        recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
        for evaluation.
        size: A number of rotations to be included in the grid.  The nearest grid
        size in log space is returned.

    Returns:
        (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
    """
    import healpy as hp

    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size / 72.0) / np.log(8.0))), 0)
    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    # Take these points on the sphere and
    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)
    polars = np.arccos(s2_points[:, 2])
    grid_rots_mats = []
    for tilt in tilts:
        # Build up the rotations from Euler angles, zyz format
        rot_mats = euler_to_so3(
            np.stack([azimuths, np.zeros(number_pix), np.zeros(number_pix)], 1)
        )
        rot_mats = rot_mats @ euler_to_so3(
            np.stack([np.zeros(number_pix), np.zeros(number_pix), polars], 1)
        )
        rot_mats = rot_mats @ torch.unsqueeze(
            euler_to_so3(np.array([tilt, 0.0, 0.0])), 0
        )
        grid_rots_mats.append(rot_mats)

    grid_rots_mats = np.concatenate(grid_rots_mats, 0)
    return grid_rots_mats
