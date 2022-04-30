"""
Symmetric Solids 1.0.0 torch dataset. Only supports single object.
"""
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from implicit_pdf.cfg import TrainConfig


class SymSolDataset(Dataset):
    """Sample torch Dataset to be used with torch DataLoader."""

    def __init__(
        self,
        split="train",
        cfg=TrainConfig(),
    ):
        assert split in {"train", "test"}
        self.cfg = cfg
        self.root = Path(getattr(self.cfg.data, f"{split}_root")).expanduser()
        self.img_root = self.root / "images"
        self.img_paths = [x for x in self.img_root.iterdir() if "tet_" in x.name]
        self.img_paths = sorted(self.img_paths)
        self.labels = np.load(self.root / "rotations.npz")["tet"]
        self.labels = torch.from_numpy(self.labels)
        assert len(self.img_paths) == self.labels.shape[0]

        # set dataset length
        if self.cfg.data.length == -1:
            self.length = self.labels.shape[0]
        else:
            assert self.cfg.data.length <= self.labels.shape[0]
            self.length = self.cfg.data.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # ignore all symmetric rotations, take only first
        so3 = self.labels[idx][0]
        # load image as np.uint8 shape (28, 28)
        x = Image.open(img_path)
        x = np.array(x)
        # convert to [0, 1.0] torch.float32, and normalize
        transform = transforms.Compose([transforms.ToTensor()])
        x = transform(x)

        return x, so3


if __name__ == "__main__":
    """Test Dataset"""
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor

    train_data = SymSolDataset(split="train")
    test_data = SymSolDataset(split="test")
    # fmt: off
    import ipdb; ipdb.set_trace(context=30)  # noqa
    # fmt: on
    print(train_data[0])
    print(test_data[0])
