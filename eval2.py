import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np

from mask_utils import interpolate_masks
from dataset import TripletDataset

# Importamos las utilidades semánticas que ya existen en tu proyecto
from semantic_metrics import compute_confusion_matrix, compute_metrics_from_confusion
from segmentador_utils import DEFAULT_EVAL_CLASSES, DEFAULT_GT_IGNORED_CLASS, get_valid_pixels_without_class

MASK_COLORS = ["#1f77b4", "#17becf", "#8c564b", "#7f7f7f", "#2ca02c", "#d62728"]
MASK_CMAP = mcolors.ListedColormap(MASK_COLORS)
MASK_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], MASK_CMAP.N)

class TripletDataset(Dataset):
    def __init__(self, base_dataset, scan_to_indices):
        self.base_dataset = base_dataset
        self.triplets = []

        for scan_id, indices in scan_to_indices.items():
            sorted_indices = sorted(indices)
            for i in range(1, len(sorted_indices) - 1):
                self.triplets.append(
                    (sorted_indices[i - 1], sorted_indices[i], sorted_indices[i + 1])
                )

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        idx_prev, idx_mid, idx_next = self.triplets[idx]
        return {
            "prev": self.base_dataset[idx_prev],
            "mid": self.base_dataset[idx_mid],
            "next": self.base_dataset[idx_next]
        }

def save_visual_comparison_interp(prev_mask, real_mid_mask, interp_mid_mask, next_mask, prev_img, real_mid_img, next_img, idx, folder, img_path="unknown"):
    prev_vis = prev_mask.cpu().numpy()
    real_vis = real_mid_mask.cpu().numpy()  
    interp_vis = interp_mid_mask.cpu().numpy()
    next_vis = next_mask.cpu().numpy()

    prev_img_vis = prev_img.cpu().numpy()[0]
    real_img_vis = real_mid_img.cpu().numpy()[0]
    next_img_vis = next_img.cpu().numpy()[0]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # Fila 0: Máscaras
    axes[0, 0].imshow(prev_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0, 0].set_title("Máscara previa")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(real_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0, 1].set_title("GT máscara real (medio)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(interp_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0, 2].set_title("Máscara interpolada")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(next_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0, 3].set_title("Máscara siguiente")
    axes[0, 3].axis("off")

    # Fila 1: Imágenes GT OCT correspondientes
    axes[1, 0].imshow(prev_img_vis, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("OCT previo")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(real_img_vis, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("GT OCT real (medio)")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off") 

    axes[1, 3].imshow(next_img_vis, cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("OCT siguiente")
    axes[1, 3].axis("off")

    fig.suptitle(f"Imagen: {img_path}", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"eval2_comparativa_{idx}.png"))
    plt.close()

@torch.no_grad()
def run_eval_2(full_dataset, test_scan_to_indices, device, base_results_dir, test_batch_size=4, shared_example_paths=None, save_every_n_batches=10):
    print("\n--- Iniciando Eval 2: Interpolación GT vs GT Real (Semántica) ---")

    if shared_example_paths is None:
        shared_example_paths = set()

    triplet_dataset = TripletDataset(full_dataset, test_scan_to_indices)
    test_loader = DataLoader(
        triplet_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )

    results_dir = base_results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Matriz de confusión para acumular los resultados de todas las imágenes
    num_classes = 6
    total_confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    pbar = tqdm(test_loader, desc="Eval 2 (Semántica)")
    for i, batch in enumerate(pbar):
        mask_prev = batch["prev"]["mask"].to(device)
        real_mid_mask = batch["mid"]["mask"].to(device)
        mask_next = batch["next"]["mask"].to(device)

        # 1. Interpolar matemáticamente
        interp_mid_mask = interpolate_masks(mask_prev, mask_next)

        # 2. Generar máscara de validación (ignorar fondo si es necesario)
        valid_mask = get_valid_pixels_without_class(
            real_mid_mask,
            class_id_to_ignore=DEFAULT_GT_IGNORED_CLASS,
        )

        # 3. Calcular matriz de confusión del batch
        batch_confusion = compute_confusion_matrix(
            pred_mask=interp_mid_mask.cpu(),
            gt_mask=real_mid_mask.cpu(),
            num_classes=num_classes,
            valid_mask=valid_mask.cpu(),
        )
        total_confusion += batch_confusion

        img_path = batch["mid"]["meta"]["image_path"][0] if "meta" in batch["mid"] and "image_path" in batch["mid"]["meta"] else "unknown"

        should_save = (save_every_n_batches is not None and save_every_n_batches > 0 and i % save_every_n_batches == 0)
        if img_path in shared_example_paths:
            should_save = True

        if should_save:
            save_visual_comparison_interp(
                mask_prev[0],         
                real_mid_mask[0],     
                interp_mid_mask[0],   
                mask_next[0],   
                batch["prev"]["image"][0],
                batch["mid"]["image"][0],
                batch["next"]["image"][0],
                i,
                results_dir,
                img_path
            )

    # 4. Calcular métricas globales al final
    global_metrics = compute_metrics_from_confusion(
        confusion_matrix=total_confusion,
        eval_classes=DEFAULT_EVAL_CLASSES,
    )

    eval2_pixel_acc = global_metrics["pixel_accuracy"]
    eval2_dice = global_metrics["mean_dice"]
    eval2_iou = global_metrics["mean_iou"]

    report_str = f"EVAL 2 - Pixel Acc: {eval2_pixel_acc:.6f} | Dice: {eval2_dice:.6f} | IoU: {eval2_iou:.6f}"
    print("\n" + report_str)

    return eval2_pixel_acc, eval2_dice, eval2_iou