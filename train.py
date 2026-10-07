from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import MoNuSegDataset
from src.metrics import (
    dice_score,
    iou_score,
    soft_dice_loss,
)
from src.model import UNet


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data/raw/MoNuSeg 2018 Training Data"
)

IMAGE_DIR = DATA_ROOT / "Tissue Images_256"
MASK_DIR = DATA_ROOT / "mask_tif_256"

MANIFEST = (
    PROJECT_ROOT
    / "data/split_manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str):
    filenames = []

    with MANIFEST.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == split:
                filenames.append(
                    row["filename"]
                )

    return filenames


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        batch_size = images.size(0)

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            logits = model(images)

            loss_bce = bce(
                logits,
                masks,
            )

            loss_dice = soft_dice_loss(
                logits,
                masks,
            )

            loss = (
                loss_bce
                + loss_dice
            )

            if training:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            dice = dice_score(
                logits,
                masks,
            )

            iou = iou_score(
                logits,
                masks,
            )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_dice += (
            dice.item()
            * batch_size
        )

        total_iou += (
            iou.item()
            * batch_size
        )

        total_samples += batch_size

    return {
        "loss":
            total_loss
            / total_samples,

        "dice":
            total_dice
            / total_samples,

        "iou":
            total_iou
            / total_samples,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--base",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

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

    train_files = load_split(
        "train"
    )

    val_files = load_split(
        "val"
    )

    print(
        "Train patches:",
        len(train_files),
    )

    print(
        "Val patches  :",
        len(val_files),
    )

    train_ds = MoNuSegDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        filenames=train_files,
        augment=True,
    )

    val_ds = MoNuSegDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        filenames=val_files,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=args.base,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_dice = -1.0
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
        )

        row = {
            "epoch": epoch,

            "train_loss":
                train_metrics["loss"],

            "train_dice":
                train_metrics["dice"],

            "train_iou":
                train_metrics["iou"],

            "val_loss":
                val_metrics["loss"],

            "val_dice":
                val_metrics["dice"],

            "val_iou":
                val_metrics["iou"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"dice={train_metrics['dice']:.4f} "
            f"iou={train_metrics['iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} "
            f"iou={val_metrics['iou']:.4f}"
        )

        if (
            val_metrics["dice"]
            > best_val_dice
        ):
            best_val_dice = (
                val_metrics["dice"]
            )

            torch.save(
                model.state_dict(),
                OUTPUT_DIR
                / "best_model.pt",
            )

            print(
                "  -> saved new best model"
            )

    with (
        OUTPUT_DIR
        / "history.json"
    ).open("w") as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    summary = {
        "best_val_dice":
            best_val_dice,

        "epochs":
            args.epochs,

        "train_patches":
            len(train_ds),

        "val_patches":
            len(val_ds),
    }

    with (
        OUTPUT_DIR
        / "training_summary.json"
    ).open("w") as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "Best validation Dice:",
        f"{best_val_dice:.4f}",
    )

    print(
        "Checkpoint:",
        OUTPUT_DIR / "best_model.pt",
    )


if __name__ == "__main__":
    main()
