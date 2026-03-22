
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - Dice Score
    Dice Score = 2|X ∩ Y| / (|X| + |Y|)

    smooth=1 prevents division by zero when both
    prediction and target are all zeros (empty mask).
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.view(-1)      # flatten to 1D
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    BCE  : penalizes per-pixel mistakes
    Dice : penalizes poor mask overlap (handles class imbalance)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce       = nn.BCELoss()
        self.dice      = DiceLoss()
        self.bce_w     = bce_weight
        self.dice_w    = dice_weight

    def forward(self, pred, target):
        bce_loss  = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_w * bce_loss + self.dice_w * dice_loss


# ── Metrics (no gradients needed — evaluation only) ──────────────────────────

def dice_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Dice Score for evaluation. Thresholds predictions to binary
    before computing — unlike Dice Loss which uses raw probabilities.
    """
    pred   = (pred > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Intersection over Union (Jaccard Index).
    IoU = |X ∩ Y| / |X ∪ Y|
        = intersection / (pred + target - intersection)
    """
    pred   = (pred > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) /
            (union + smooth)).item()
