import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from ignite.metrics import SSIM
from dataset import OCT5KDataset
import numpy as np
import torch

from mask_utils import interpolate_masks
from dataset import TripletDataset


MASK_COLORS = ["#1f77b4", "#17becf", "#8c564b", "#7f7f7f", "#2ca02c", "#d62728"]
MASK_CMAP = mcolors.ListedColormap(MASK_COLORS)
MASK_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], MASK_CMAP.N)

@torch.no_grad()
def run_eval_1(model, full_dataset, test_scan_to_indices, device, base_results_dir, test_batch_size=4, shared_example_paths=None, save_every_n_batches=10):
    print("\n--- Iniciando Eval 1: Síntesis desde GT interpolado vs GT Real ---")

    if shared_example_paths is None:
        shared_example_paths = set()

    triplet_dataset = TripletDataset(full_dataset, test_scan_to_indices)
    test_loader = DataLoader(triplet_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    results_dir = base_results_dir
    os.makedirs(results_dir, exist_ok=True)

    ssim_metric = SSIM(data_range=1.0, device=device)
    mse_fn = nn.MSELoss()
    l1_fn = nn.L1Loss()
    total_mse, total_l1, samples_count = 0, 0, 0

    pbar = tqdm(test_loader, desc="Eval 1")
    for i, batch in enumerate(pbar):
        mask_prev = batch["prev"]["mask"].to(device)
        mask_next = batch["next"]["mask"].to(device)
        real_mid_mask = batch["mid"]["mask"].to(device) # Cargar la máscara real
        real_mid_img = batch["mid"]["image"].to(device)

        interp_mid_mask = interpolate_masks(mask_prev, mask_next)
        cond_interp = OCT5KDataset.mask_to_one_hot(interp_mid_mask, num_classes=6).float()

        raw_output = model(cond_interp)
        pred_mid_img = torch.sigmoid(raw_output.float())

        ssim_metric.update((pred_mid_img, real_mid_img))
        n = real_mid_img.size(0)
        total_mse += mse_fn(pred_mid_img, real_mid_img).item() * n
        total_l1 += l1_fn(pred_mid_img, real_mid_img).item() * n
        samples_count += n

        img_path = batch["mid"]["meta"]["image_path"][0]
        should_save = (save_every_n_batches is not None and save_every_n_batches > 0 and i % save_every_n_batches == 0)
        if img_path in shared_example_paths:
            should_save = True

        if should_save:
            # Guarda periódicamente para asegurar ejemplos, y también cuando coincide una ruta compartida.
            save_visual_comparison(interp_mid_mask[0], real_mid_mask[0], real_mid_img[0], pred_mid_img[0], i, results_dir, img_path)

    avg_ssim = ssim_metric.compute()
    avg_mse = total_mse / samples_count
    avg_l1 = total_l1 / samples_count

    report_str = f"EVAL 1 - SSIM: {avg_ssim:.6f} | MSE: {avg_mse:.6f} | L1: {avg_l1:.6f}"
    print("\n" + report_str)

    return avg_ssim, avg_mse, avg_l1


def save_visual_comparison(interp_mask, real_mask, real_img, pred_img, idx, folder, img_path):
    mask_interp_vis = interp_mask.cpu().numpy()
    mask_real_vis = real_mask.cpu().numpy()
    
    real_vis = real_img.cpu().numpy()[0]
    pred_vis = pred_img.cpu().numpy()[0]

    # Ampliado a 4 columnas
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    axes[0].imshow(mask_interp_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0].set_title("GT Máscara Interpolada")
    axes[0].axis("off")

    axes[1].imshow(mask_real_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[1].set_title("GT Máscara Real (Medio)")
    axes[1].axis("off")

    axes[2].imshow(real_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("GT OCT Real (Medio)")
    axes[2].axis("off")

    axes[3].imshow(pred_vis, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Síntesis Generada")
    axes[3].axis("off")

    fig.suptitle(f"Imagen: {img_path}", fontsize=10, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"comp_eval1_{idx}.png"))
    plt.close()