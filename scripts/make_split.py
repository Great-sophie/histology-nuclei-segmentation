from pathlib import Path
import sys
import csv
import random

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import source_case_from_filename


IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/MoNuSeg 2018 Training Data/Tissue Images_256"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data/raw/MoNuSeg 2018 Training Data/mask_tif_256"
)

OUTPUT = PROJECT_ROOT / "data/split_manifest.csv"

SEED = 42


def main():
    filenames = sorted(
        p.name for p in IMAGE_DIR.glob("*.tif")
    )

    print(f"Total patches: {len(filenames)}")

    if len(filenames) == 0:
        raise RuntimeError(
            f"No TIFF files found in:\n{IMAGE_DIR}"
        )

    # --------------------------------------------------
    # Check image-mask pairing
    # --------------------------------------------------
    missing_masks = []

    for name in filenames:
        if not (MASK_DIR / name).exists():
            missing_masks.append(name)

    if missing_masks:
        raise RuntimeError(
            f"{len(missing_masks)} masks missing. "
            f"Example: {missing_masks[0]}"
        )

    print("Image-mask pairing: OK")

    # --------------------------------------------------
    # Group patches by source image/case
    # --------------------------------------------------
    case_to_files = {}

    for name in filenames:
        case_id = source_case_from_filename(name)

        case_to_files.setdefault(
            case_id,
            []
        ).append(name)

    cases = sorted(case_to_files.keys())

    print(f"Source cases: {len(cases)}")

    # --------------------------------------------------
    # Case-level split
    # --------------------------------------------------
    rng = random.Random(SEED)
    rng.shuffle(cases)

    n_cases = len(cases)

    n_train = int(n_cases * 0.70)
    n_val = int(n_cases * 0.15)

    train_cases = set(
        cases[:n_train]
    )

    val_cases = set(
        cases[n_train:n_train + n_val]
    )

    test_cases = set(
        cases[n_train + n_val:]
    )

    # --------------------------------------------------
    # Safety checks: no case overlap
    # --------------------------------------------------
    assert train_cases.isdisjoint(val_cases)
    assert train_cases.isdisjoint(test_cases)
    assert val_cases.isdisjoint(test_cases)

    # --------------------------------------------------
    # Build manifest
    # --------------------------------------------------
    rows = []

    for case_id, files in case_to_files.items():

        if case_id in train_cases:
            split = "train"

        elif case_id in val_cases:
            split = "val"

        else:
            split = "test"

        for filename in files:
            rows.append(
                {
                    "filename": filename,
                    "case_id": case_id,
                    "split": split,
                }
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "case_id",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    print()
    print("Cases")
    print("-----")
    print("train:", len(train_cases))
    print("val  :", len(val_cases))
    print("test :", len(test_cases))

    print()
    print("Patches")
    print("-------")

    for split in [
        "train",
        "val",
        "test",
    ]:
        n = sum(
            row["split"] == split
            for row in rows
        )

        print(
            f"{split:5s}: {n}"
        )

    print()
    print("No case leakage: YES")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
