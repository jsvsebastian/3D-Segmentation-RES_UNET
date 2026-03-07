# src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss para segmentación binaria.
    Insensible al desbalance de clases.
    
    Dice = 1 - (2 * |Pred ∩ GT|) / (|Pred| + |GT| + smooth)
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred:   logits (B, 1, D, H, W) — sin sigmoid todavía
        # target: máscara binaria (B, 1, D, H, W)
        pred = torch.sigmoid(pred)

        # Aplanar espacialmente
        pred_flat   = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
               pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Pérdida combinada: Dice Loss + Binary Cross Entropy.
    
    - BCE:  penaliza cada vóxel individualmente
    - Dice: penaliza el solapamiento global, insensible al desbalance
    
    Loss = dice_weight * DiceLoss + bce_weight * BCE
    """
    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1e-5):
        super().__init__()
        self.dice     = DiceLoss(smooth=smooth)
        self.bce      = nn.BCEWithLogitsLoss()
        self.dw       = dice_weight
        self.bw       = bce_weight

    def forward(self, pred, target):
        loss_dice = self.dice(pred, target)
        loss_bce  = self.bce(pred, target)
        return self.dw * loss_dice + self.bw * loss_bce, loss_dice, loss_bce


def dice_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    Métrica Dice Score para evaluación (no para backprop).
    Aplica threshold para obtener máscara binaria.
    
    Returns: float entre 0 y 1
    """
    with torch.no_grad():
        pred_bin    = (torch.sigmoid(pred) > threshold).float()
        pred_flat   = pred_bin.view(pred_bin.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        score = (2.0 * intersection + smooth) / (
                pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)

        return score.mean().item()