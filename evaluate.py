from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import MoNuSegDataset, source_case_from_filename
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

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_test_files():
    filenames = []

    with MANIFEST.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == "test":
                filenames.append(row["filename"])

    return filenames


def per_sample_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7,
):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (1, 2, 3)

    intersection = (
        preds * targets
    ).sum(dim=dims)

    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)

    dice = (
        2 * intersection + eps
    ) / (
        pred_sum + target_sum + eps
    )

    union = (
        pred_sum
        + target_sum
        - intersection
    )

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return dice, iou


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    test_files = load_test_files()

    print(
        "Test patches:",
        len(test_files)
    )

    test_ds = MoNuSegDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        filenames=test_files,
        augment=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=16,
    ).to(device)

    state = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(state)
    model.eval()

    rows = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)

            dice, iou = per_sample_metrics(
                logits,
                masks,
            )

            for i, filename in enumerate(
                batch["filename"]
            ):
                rows.append(
                    {
                        "filename": filename,
                        "case_id":
                            source_case_from_filename(
                                filename
                            ),
                        "dice":
                            float(dice[i].item()),
                        "iou":
                            float(iou[i].item()),
                    }
                )

    patch_dice = np.array(
        [r["dice"] for r in rows]
    )

    patch_iou = np.array(
        [r["iou"] for r in rows]
    )

    # -----------------------------
    # Case-level aggregation
    # -----------------------------
    case_results = defaultdict(
        lambda: {
            "dice": [],
            "iou": [],
        }
    )

    for row in rows:
        case_results[row["case_id"]][
            "dice"
        ].append(row["dice"])

        case_results[row["case_id"]][
            "iou"
        ].append(row["iou"])

    case_rows = []

    for case_id, values in case_results.items():

        case_rows.append(
            {
                "case_id": case_id,
                "mean_dice":
                    float(
                        np.mean(values["dice"])
                    ),
                "mean_iou":
                    float(
                        np.mean(values["iou"])
                    ),
                "n_patches":
                    len(values["dice"]),
            }
        )

    case_dice = np.array(
        [r["mean_dice"] for r in case_rows]
    )

    case_iou = np.array(
        [r["mean_iou"] for r in case_rows]
    )

    summary = {
        "n_test_patches": len(rows),
        "n_test_cases": len(case_rows),

        "patch_mean_dice":
            float(patch_dice.mean()),

        "patch_median_dice":
            float(np.median(patch_dice)),

        "patch_std_dice":
            float(patch_dice.std()),

        "patch_mean_iou":
            float(patch_iou.mean()),

        "patch_median_iou":
            float(np.median(patch_iou)),

        "case_mean_dice":
            float(case_dice.mean()),

        "case_mean_iou":
            float(case_iou.mean()),
    }

    # -----------------------------
    # Save outputs
    # -----------------------------
    with (
        OUTPUT_DIR
        / "test_patch_metrics.csv"
    ).open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "case_id",
                "dice",
                "iou",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    with (
        OUTPUT_DIR
        / "test_case_metrics.csv"
    ).open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "mean_dice",
                "mean_iou",
                "n_patches",
            ],
        )

        writer.writeheader()
        writer.writerows(case_rows)

    with (
        OUTPUT_DIR
        / "test_metrics.json"
    ).open("w") as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("TEST RESULTS")
    print("============")

    print(
        "Patches:",
        summary["n_test_patches"],
    )

    print(
        "Cases  :",
        summary["n_test_cases"],
    )

    print()

    print(
        "Patch mean Dice :",
        f'{summary["patch_mean_dice"]:.4f}'
    )

    print(
        "Patch mean IoU  :",
        f'{summary["patch_mean_iou"]:.4f}'
    )

    print(
        "Case mean Dice  :",
        f'{summary["case_mean_dice"]:.4f}'
    )

    print(
        "Case mean IoU   :",
        f'{summary["case_mean_iou"]:.4f}'
    )


if __name__ == "__main__":
    main()
