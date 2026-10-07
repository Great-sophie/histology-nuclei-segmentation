import torch


def dice_score(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7,
):
    probs = torch.sigmoid(logits)

    preds = (
        probs >= threshold
    ).float()

    dims = tuple(
        range(1, preds.ndim)
    )

    intersection = (
        preds * targets
    ).sum(dim=dims)

    denominator = (
        preds.sum(dim=dims)
        + targets.sum(dim=dims)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return dice.mean()


def iou_score(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7,
):
    probs = torch.sigmoid(logits)

    preds = (
        probs >= threshold
    ).float()

    dims = tuple(
        range(1, preds.ndim)
    )

    intersection = (
        preds * targets
    ).sum(dim=dims)

    union = (
        preds.sum(dim=dims)
        + targets.sum(dim=dims)
        - intersection
    )

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return iou.mean()


def soft_dice_loss(
    logits,
    targets,
    eps=1e-7,
):
    probs = torch.sigmoid(logits)

    dims = tuple(
        range(1, probs.ndim)
    )

    intersection = (
        probs * targets
    ).sum(dim=dims)

    denominator = (
        probs.sum(dim=dims)
        + targets.sum(dim=dims)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()
