from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class MoNuSegDataset(Dataset):
    """
    MoNuSeg 256x256 H&E nuclei segmentation dataset.

    Image:
        RGB uint8 -> float32
        [H, W, 3] -> [3, H, W]
        value range: 0..255 -> 0..1

    Mask:
        grayscale uint8 -> float32
        [H, W] -> [1, H, W]
        {0, 255} -> {0, 1}
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        filenames: Sequence[str] | None = None,
        augment: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment

        if filenames is None:
            filenames = sorted(
                p.name for p in self.image_dir.glob("*.tif")
            )

        self.filenames = list(filenames)

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No TIFF files found in {self.image_dir}"
            )

        # Verify every image has a corresponding mask.
        missing_masks = [
            name
            for name in self.filenames
            if not (self.mask_dir / name).exists()
        ]

        if missing_masks:
            raise RuntimeError(
                f"{len(missing_masks)} masks are missing. "
                f"Example: {missing_masks[0]}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def _augment(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):
        # Same geometric transformation must be applied
        # to BOTH image and mask.

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random 90-degree rotations.
        k = random.randint(0, 3)

        if k:
            angle = 90 * k
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask

    def __getitem__(self, idx: int):
        name = self.filenames[idx]

        image_path = self.image_dir / name
        mask_path = self.mask_dir / name

        # Real H&E image
        image = Image.open(image_path).convert("RGB")

        # Binary mask
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._augment(image, mask)

        image = np.asarray(image, dtype=np.float32) / 255.0

        mask = np.asarray(mask, dtype=np.uint8)

        # Robust binary conversion:
        # anything > 0 becomes foreground.
        mask = (mask > 0).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # HW -> 1HW
        mask = mask[None, ...]

        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())

        return {
            "image": image,
            "mask": mask,
            "filename": name,
        }


def source_case_from_filename(filename: str) -> str:
    """
    Convert patch filename:

    TCGA-18-5592-01Z-00-DX1-x0_y256_w256_h256-mask.tif

    into source case/image ID:

    TCGA-18-5592-01Z-00-DX1
    """
    return filename.rsplit("-x", 1)[0]
