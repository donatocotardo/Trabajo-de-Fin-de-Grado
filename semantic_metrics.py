import math
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
import torch


def _to_long_tensor(x) -> torch.Tensor:
    """
    Convierte una máscara numpy/tensor a torch.Tensor long en CPU.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().long()

    return torch.from_numpy(np.asarray(x)).long()


def compute_confusion_matrix(
    pred_mask,
    gt_mask,
    num_classes: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Matriz de confusión con filas = GT y columnas = predicción.

    Solo se consideran los píxeles válidos.
    """
    pred = _to_long_tensor(pred_mask)
    gt = _to_long_tensor(gt_mask)

    if pred.shape != gt.shape:
        raise ValueError(
            f"pred_mask y gt_mask deben tener la misma forma. "
            f"Recibido pred={tuple(pred.shape)}, gt={tuple(gt.shape)}"
        )

    if valid_mask is None:
        valid = torch.ones_like(gt, dtype=torch.bool)
    else:
        valid = valid_mask.detach().cpu().bool()

    valid = valid & (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)

    gt_valid = gt[valid].view(-1)
    pred_valid = pred[valid].view(-1)

    if gt_valid.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long)

    encoded = gt_valid * num_classes + pred_valid

    cm = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    return cm.long()


def compute_metrics_from_confusion(
    confusion_matrix: torch.Tensor,
    eval_classes: Sequence[int],
) -> Dict[str, Any]:
    """
    Calcula pixel accuracy, Dice e IoU desde una matriz de confusión.

    confusion_matrix:
        filas = GT
        columnas = predicción
    """
    cm = confusion_matrix.detach().cpu().double()

    total_valid = cm.sum().item()
    correct = torch.diag(cm).sum().item()

    if total_valid > 0:
        pixel_accuracy = correct / total_valid
    else:
        pixel_accuracy = float("nan")

    per_class = {}

    dice_values = []
    iou_values = []

    for cls in eval_classes:
        cls = int(cls)

        tp = cm[cls, cls].item()
        fp = cm[:, cls].sum().item() - tp
        fn = cm[cls, :].sum().item() - tp

        support_gt = cm[cls, :].sum().item()
        support_pred = cm[:, cls].sum().item()

        dice_den = (2.0 * tp + fp + fn)
        iou_den = (tp + fp + fn)

        if dice_den > 0:
            dice = (2.0 * tp) / dice_den
            dice_values.append(dice)
        else:
            dice = float("nan")

        if iou_den > 0:
            iou = tp / iou_den
            iou_values.append(iou)
        else:
            iou = float("nan")

        per_class[cls] = {
            "dice": dice,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support_gt": support_gt,
            "support_pred": support_pred,
        }

    mean_dice = float(np.mean(dice_values)) if len(dice_values) > 0 else float("nan")
    mean_iou = float(np.mean(iou_values)) if len(iou_values) > 0 else float("nan")

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "per_class": per_class,
        "total_valid_pixels": total_valid,
        "correct_pixels": correct,
        "confusion_matrix": confusion_matrix.long(),
    }
