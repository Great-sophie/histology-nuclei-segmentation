from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import MoNuSegDataset
from src.model import UNet


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data/raw/MoNuSeg 2018 Training Data"
)

IMAGE_DIR = DATA_ROOT / "Tissue Images_256"
MASK_DIR = DATA_ROOT / "mask_tif_256"

MANIFEST = PROJECT_ROOT / "data/split_manifest.csv"

CHECKPOINT = PROJECT_ROOT / "outputs/best_model.pt"

OUTPUT = (
    PROJECT_ROOT
    / "outputs/test_predictions.png"
)


def load_test_files():
    files = []

    with MANIFEST.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == "test":
                files.append(row["filename"])

    return files


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    test_files = load_test_files()

    # Use first 5 test examples
    selected = test_files[:5]

    ds = MoNuSegDataset(
        IMAGE_DIR,
        MASK_DIR,
        filenames=selected,
        augment=False,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=16,
    ).to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
        )
    )

    model.eval()

    fig, axes = plt.subplots(
        len(ds),
        4,
        figsize=(12, 3 * len(ds)),
    )

    with torch.no_grad():

        for row in range(len(ds)):

            sample = ds[row]

            image = sample["image"]

            mask = sample["mask"][0]

            logits = model(
                image.unsqueeze(0).to(device)
            )

            prob = torch.sigmoid(
                logits
            )[0, 0]

            pred = (
                prob >= 0.5
            ).float().cpu()

            image_np = (
                image
                .permute(1, 2, 0)
                .numpy()
            )

            mask_np = mask.numpy()
            pred_np = pred.numpy()

            axes[row, 0].imshow(
                image_np
            )

            axes[row, 0].set_title(
                "H&E"
            )

            axes[row, 1].imshow(
                mask_np,
                cmap="gray",
            )

            axes[row, 1].set_title(
                "Ground Truth"
            )

            axes[row, 2].imshow(
                pred_np,
                cmap="gray",
            )

            axes[row, 2].set_title(
                "Prediction"
            )

            axes[row, 3].imshow(
                image_np
            )

            axes[row, 3].imshow(
                pred_np,
                alpha=0.35,
            )

            axes[row, 3].set_title(
                "Overlay"
            )

            for col in range(4):
                axes[row, col].axis(
                    "off"
                )

    plt.tight_layout()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        OUTPUT
    )


if __name__ == "__main__":
    main()
