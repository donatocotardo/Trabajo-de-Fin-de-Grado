import os
import hashlib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
from functools import partial
import segmentation_models_pytorch as smp
from ignite.metrics import SSIM

# --- IMPORTS PARA TTA ---
from torchvision.transforms import v2
from torchvision import tv_tensors

from dataset import OCT5KDataset
from eval1 import run_eval_1
from eval2 import run_eval_2
from eval3 import run_eval_3
from eval4 import run_eval_4
from utils import plot_loss_curves


MASK_COLORS = ["#1f77b4", "#17becf", "#8c564b", "#7f7f7f", "#2ca02c", "#d62728"]
MASK_CMAP = mcolors.ListedColormap(MASK_COLORS)
MASK_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], MASK_CMAP.N)

# ===================================================================
# TEST-TIME AUGMENTATION GEOMÉTRICO Y DETERMINISTA
# ===================================================================
def configure_reproducibility(seed: int) -> None:
    """Configura las fuentes de aleatoriedad generales de la evaluación."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Evita que cuDNN seleccione algoritmos distintos entre ejecuciones.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _tensor_sample_key(img: torch.Tensor, mask: torch.Tensor) -> str:
    """Genera una clave estable cuando no se dispone de la ruta de la imagen."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(img.detach().cpu().contiguous().numpy().tobytes())
    digest.update(mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sample_tta_seed(base_seed: int, sample_key: str) -> int:
    """Obtiene una semilla reproducible y distinta para cada imagen."""
    key_digest = hashlib.blake2b(str(sample_key).encode("utf-8"), digest_size=8).digest()
    key_value = int.from_bytes(key_digest, byteorder="little", signed=False)
    return (int(base_seed) + key_value) % (2**31 - 1)


def _apply_single_geometric_tta(
    img: torch.Tensor,
    mask: torch.Tensor,
    elastic_transform: v2.ElasticTransform,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica una combinación aleatoria, pero reproducible, a una única imagen."""
    _, _, h, w = img.shape
    cuda_devices = [img.device.index] if img.is_cuda and img.device.index is not None else []

    # fork_rng restaura el estado global al salir, por lo que una muestra no altera
    # la transformación que recibirá otra muestra.
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if img.is_cuda:
            torch.cuda.manual_seed_all(seed)

        # 1. Flip horizontal (p=0.5)
        if torch.rand(()).item() < 0.5:
            img = torch.flip(img, dims=[3])
            mask = torch.flip(mask, dims=[2])

        # 2. Deformación elástica (p=0.5)
        if torch.rand(()).item() < 0.5:
            img_tv = tv_tensors.Image(img)
            mask_tv = tv_tensors.Mask(mask.unsqueeze(1))
            img_tv, mask_tv = elastic_transform(img_tv, mask_tv)
            img = img_tv.as_subclass(torch.Tensor)
            mask = mask_tv.as_subclass(torch.Tensor).squeeze(1)

        # 3. Random resized crop (p=0.5)
        if torch.rand(()).item() < 0.5:
            scale = 0.80 + 0.20 * torch.rand(()).item()
            crop_h = max(1, int(h * scale))
            crop_w = max(1, int(w * scale))
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()

            img = img[:, :, top:top + crop_h, left:left + crop_w]
            mask = mask[:, top:top + crop_h, left:left + crop_w]

            img = torch.nn.functional.interpolate(
                img,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(),
                size=(h, w),
                mode="nearest",
            ).squeeze(1).long()

    return img, mask


def apply_geometric_tta(
    img: torch.Tensor,
    mask: torch.Tensor,
    elastic_transform: v2.ElasticTransform,
    base_seed: int = 12345,
    sample_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica TTA determinista por muestra.

    La semilla de cada muestra se obtiene combinando ``base_seed`` con una clave
    estable de la imagen. Por ello, una misma imagen recibe siempre exactamente
    el mismo flip, deformación elástica y recorte, aunque se evalúe otro modelo,
    se cambie el batch size o se ejecute el script en otro momento.
    """
    if img.ndim != 4 or mask.ndim != 3:
        raise ValueError(
            f"Dimensiones no válidas: img={tuple(img.shape)}, mask={tuple(mask.shape)}. "
            "Se esperaban [B,C,H,W] y [B,H,W]."
        )

    batch_size = img.shape[0]
    if sample_keys is not None and len(sample_keys) != batch_size:
        raise ValueError(
            f"sample_keys contiene {len(sample_keys)} elementos, pero el batch tiene "
            f"{batch_size} muestras."
        )

    transformed_images = []
    transformed_masks = []

    for sample_idx in range(batch_size):
        img_sample = img[sample_idx:sample_idx + 1]
        mask_sample = mask[sample_idx:sample_idx + 1]

        if sample_keys is None:
            sample_key = _tensor_sample_key(img_sample, mask_sample)
        else:
            sample_key = str(sample_keys[sample_idx])

        seed = _sample_tta_seed(base_seed, sample_key)
        img_sample, mask_sample = _apply_single_geometric_tta(
            img=img_sample,
            mask=mask_sample,
            elastic_transform=elastic_transform,
            seed=seed,
        )
        transformed_images.append(img_sample)
        transformed_masks.append(mask_sample)

    return torch.cat(transformed_images, dim=0), torch.cat(transformed_masks, dim=0)
# ===================================================================


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def get_scan_id(sample):
    path = Path(sample["image"])
    return path.parent.name


def build_test_run_name(checkpoint_path, checkpoint, test_batch_size, use_tta=False, tta_seed=12345):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    tta_tag = f"_TTA_seed{tta_seed}" if use_tta else ""

    if config:
        exp_name = config.get("name", "exp")
        loss_type = config.get("loss_type", "NA")
        lr = config.get("lr", "NA")
        train_bs = config.get("batch_size", "NA")
        epochs = config.get("epochs", "NA")

        base_name = f"ckpt_{exp_name}_loss{loss_type}_lr{lr}_bs{train_bs}_ep{epochs}"
        if "interp_weight" in config:
            base_name += f"_interpW{config['interp_weight']}"
        return f"{base_name}_testbs{test_batch_size}{tta_tag}"

    return f"{Path(checkpoint_path).stem}_testbs{test_batch_size}{tta_tag}"


def build_shared_example_paths(full_dataset, test_indices, test_batch_size, every_n_batches=10):
    selected_paths = set()
    if every_n_batches is None or every_n_batches <= 0:
        return selected_paths

    for batch_idx, subset_start in enumerate(range(0, len(test_indices), test_batch_size)):
        if batch_idx % every_n_batches != 0:
            continue
        dataset_idx = test_indices[subset_start]
        selected_paths.add(str(full_dataset.samples[dataset_idx]["image"]))

    return selected_paths


def build_reconstruction_metrics_table(all_metrics):
    header = f"| {'Tipo de Evaluación':<40} | {'SSIM':<8} | {'MSE':<8} | {'L1':<8} |"
    separator = "|" + "-" * 42 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|"
    table_lines = [
        "RESUMEN DE MÉTRICAS GLOBALES",
        "============================",
        "",
        header,
        separator,
    ]
    for row in all_metrics:
        line = (
            f"| {row['Eval Type']:<40} | "
            f"{to_float(row['SSIM']):<8.4f} | "
            f"{to_float(row['MSE']):<8.4f} | "
            f"{to_float(row['L1']):<8.4f} |"
        )
        table_lines.append(line)
    return "\n".join(table_lines)


def build_semantic_metrics_table(semantic_metrics):
    if not semantic_metrics:
        return ""
    header = (
        f"| {'Tipo de Evaluación':<42} | "
        f"{'Pixel Acc':<9} | "
        f"{'Dice':<8} | "
        f"{'IoU':<8} |"
    )
    separator = "|" + "-" * 44 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 10 + "|"
    table_lines = [
        "RESUMEN DE MÉTRICAS SEMÁNTICAS",
        "==============================",
        "",
        header,
        separator,
    ]
    for row in semantic_metrics:
        line = (
            f"| {row['Eval Type']:<42} | "
            f"{to_float(row['Pixel Acc']):<9.4f} | "
            f"{to_float(row['Dice']):<8.4f} | "
            f"{to_float(row['IoU']):<8.4f} |"
        )
        table_lines.append(line)
    return "\n".join(table_lines)


def build_activation_metrics_table(activation_metrics):
    if not activation_metrics:
        return ""
    header = (
        f"| {'Tipo de Evaluación':<36} | "
        f"{'Layer':<10} | "
        f"{'Feature MSE':<12} | "
        f"{'Feature L1':<11} | "
        f"{'Cosine':<8} |"
    )
    separator = "|" + "-" * 38 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 10 + "|"
    table_lines = [
        "RESUMEN DE ACTIVACIONES INTERMEDIAS",
        "===================================",
        "",
        header,
        separator,
    ]
    eval_name = "4. Activaciones Real vs Sintética"
    for row in activation_metrics:
        line = (
            f"| {eval_name:<36} | "
            f"{row['layer']:<10} | "
            f"{to_float(row['feature_mse']):<12.6f} | "
            f"{to_float(row['feature_l1']):<11.6f} | "
            f"{to_float(row['cosine']):<8.4f} |"
        )
        table_lines.append(line)
    return "\n".join(table_lines)


def latex_escape(text):
    text = str(text)
    replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_reconstruction_latex_table(all_metrics):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Global reconstruction and interpolation metrics.}")
    lines.append(r"\label{tab:reconstruction_metrics}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Evaluation & SSIM $\uparrow$ & MSE $\downarrow$ & L1 $\downarrow$ \\")
    lines.append(r"\hline")
    for row in all_metrics:
        eval_name = latex_escape(row["Eval Type"])
        ssim = to_float(row["SSIM"])
        mse = to_float(row["MSE"])
        l1 = to_float(row["L1"])
        lines.append(f"{eval_name} & {ssim:.4f} & {mse:.4f} & {l1:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_semantic_latex_table(semantic_metrics):
    if not semantic_metrics:
        return ""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Semantic evaluation using the external segmentation model.}")
    lines.append(r"\label{tab:semantic_metrics}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Evaluation & Pixel Acc. $\uparrow$ & Dice $\uparrow$ & IoU $\uparrow$ \\")
    lines.append(r"\hline")
    for row in semantic_metrics:
        eval_name = latex_escape(row["Eval Type"])
        pixel_acc = to_float(row["Pixel Acc"])
        dice = to_float(row["Dice"])
        iou = to_float(row["IoU"])
        lines.append(f"{eval_name} & {pixel_acc:.4f} & {dice:.4f} & {iou:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_activation_latex_table(activation_metrics):
    if not activation_metrics:
        return ""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Intermediate activation similarity between real and synthetic OCT images.}")
    lines.append(r"\label{tab:activation_metrics}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"Layer & Feature MSE $\downarrow$ & Feature L1 $\downarrow$ & Cosine $\uparrow$ \\")
    lines.append(r"\hline")
    for row in activation_metrics:
        layer = latex_escape(row["layer"])
        feature_mse = to_float(row["feature_mse"])
        feature_l1 = to_float(row["feature_l1"])
        cosine = to_float(row["cosine"])
        lines.append(f"{layer} & {feature_mse:.6f} & {feature_l1:.6f} & {cosine:.4f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_latex_metrics_report(all_metrics, semantic_metrics, activation_metrics):
    blocks = []
    blocks.append("% ========================================================")
    blocks.append("% Reconstruction / interpolation metrics")
    blocks.append("% ========================================================")
    blocks.append(build_reconstruction_latex_table(all_metrics))

    if semantic_metrics:
        blocks.append("")
        blocks.append("% ========================================================")
        blocks.append("% Semantic metrics")
        blocks.append("% ========================================================")
        blocks.append(build_semantic_latex_table(semantic_metrics))

    if activation_metrics:
        blocks.append("")
        blocks.append("% ========================================================")
        blocks.append("% Intermediate activation metrics")
        blocks.append("% ========================================================")
        blocks.append(build_activation_latex_table(activation_metrics))

    blocks.append("")
    return "\n".join(blocks)

@torch.no_grad()
def run_test(
    checkpoint_path,
    test_batch_size=4,
    save_every_n_batches=10,
    data_root=None,
    use_tta=False,
    tta_seed=12345,
):
    configure_reproducibility(tta_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_root is None:
        data_root = str(Path(__file__).resolve().parent / "OCT_5k")
    print(f"Iniciando Test en: {device}")
    print(f"Test-Time Augmentation (TTA): {'ACTIVADO' if use_tta else 'DESACTIVADO'}")
    if use_tta:
        print(f"Semilla TTA determinista: {tta_seed}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: No se encontró el archivo {checkpoint_path}")
        return

    full_dataset = OCT5KDataset(
        data_root=data_root,
        load_segmentation=True,
        return_pair=True,
    )

    scan_to_indices = defaultdict(list)

    for idx, sample in enumerate(full_dataset.samples):
        scan_to_indices[get_scan_id(sample)].append(idx)

    scan_ids = sorted(list(scan_to_indices.keys()))
    n_trainval = int(len(scan_ids) * 0.8)

    test_scans = scan_ids[n_trainval:]
    test_indices = [idx for sid in test_scans for idx in scan_to_indices[sid]]

    shared_example_paths = build_shared_example_paths(
        full_dataset=full_dataset,
        test_indices=test_indices,
        test_batch_size=test_batch_size,
        every_n_batches=save_every_n_batches,
    )

    test_scan_to_indices = {sid: scan_to_indices[sid] for sid in test_scans}

    print(f"Evaluando sobre {len(test_scans)} scans de test ({len(test_indices)} imágenes)")

    test_loader = DataLoader(
        Subset(full_dataset, test_indices),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=6,
        classes=1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"¡Éxito! Pesos cargados desde: {checkpoint_path}")

    experiment_name = build_test_run_name(
        checkpoint_path, checkpoint, test_batch_size, use_tta, tta_seed
    )

    checkpoint_train_losses = []
    if isinstance(checkpoint, dict):
        checkpoint_train_losses = list(checkpoint.get("train_losses", []))

    if checkpoint_train_losses:
        plot_loss_curves(
            checkpoint_train_losses,
            log_dir=os.path.join("results", experiment_name),
            experiment_name=f"{experiment_name} (checkpoint)",
        )
        print(f"Gráficas de pérdida guardadas en: {os.path.join('results', experiment_name, 'plots')}")
    else:
        print("Aviso: este checkpoint no contiene train_losses, así que no se puede generar la curva de pérdida.")

    results_dir = os.path.join("results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    std_results_dir = os.path.join(results_dir, "0_eval_estandar")
    os.makedirs(std_results_dir, exist_ok=True)

    model.eval()

    all_metrics = []
    semantic_metrics = []
    activation_metrics = []

    print("\n" + "-" * 50)
    print("INICIANDO EVALUACIÓN ESTÁNDAR (1 a 1)")
    print("-" * 50)

    ssim_metric = SSIM(data_range=1.0, device=device)
    mse_fn = nn.MSELoss()
    l1_fn = nn.L1Loss()

    total_mse = 0.0
    total_l1 = 0.0
    samples_count = 0

    # Inicializar el transformador elástico para TTA
    elastic = v2.ElasticTransform(alpha=[500, 5000], sigma=[50, 50], fill=10).to(device)

    pbar = tqdm(test_loader, desc="Eval Estándar")

    for i, batch in enumerate(pbar):
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        # Recuperar las rutas para que cada imagen tenga una semilla propia y estable.
        image_paths = None
        if "meta" in batch and "image_path" in batch["meta"]:
            raw_paths = batch["meta"]["image_path"]
            if isinstance(raw_paths, str):
                image_paths = [raw_paths]
            else:
                image_paths = [str(path) for path in raw_paths]

        # Aplicar exactamente la misma TTA a cada imagen en todas las ejecuciones.
        if use_tta:
            img, mask = apply_geometric_tta(
                img,
                mask,
                elastic,
                base_seed=tta_seed,
                sample_keys=image_paths,
            )

        # Generar máscara de validez (ignorar fondo clase 10 introducido por TTA)
        valid = (mask != 10).float()

        # Limitar las clases a un máximo de 5 para el one_hot (evitar fallos con el fondo 10)
        cond = OCT5KDataset.mask_to_one_hot(mask.clamp(0, 5), num_classes=6).float()
        target = img.float()

        raw_output = model(cond)
        output = torch.sigmoid(raw_output.float())

        # Aplicar la máscara de validez a las imágenes generadas y objetivo
        output_valid = output * valid.unsqueeze(1)
        target_valid = target * valid.unsqueeze(1)

        ssim_metric.update((output_valid, target_valid))

        n = cond.size(0)
        total_mse += mse_fn(output_valid, target_valid).item() * n
        total_l1 += l1_fn(output_valid, target_valid).item() * n
        samples_count += n

        img_path = image_paths[0] if image_paths else "unknown"

        if img_path in shared_example_paths:
            save_visual_comparison(
                cond[0],
                target_valid[0],
                output_valid[0],
                i,
                std_results_dir,
                img_path,
            )

    std_ssim = ssim_metric.compute()
    std_mse = total_mse / samples_count
    std_l1 = total_l1 / samples_count

    tta_label = " (TTA)" if use_tta else ""
    all_metrics.append(
        {
            "Eval Type": f"0. Estándar{tta_label} (Mask GT vs Img GT)",
            "SSIM": std_ssim,
            "MSE": std_mse,
            "L1": std_l1,
        }
    )

    # Las demás evaluaciones se ejecutan de forma normal, como solicitaste.
    eval1_results_dir = os.path.join(results_dir, "1_eval_sintesis_desde_interpolacion")
    eval2_results_dir = os.path.join(results_dir, "2_eval_interpolacion_pura")
    eval3_results_dir = os.path.join(results_dir, "3_eval_semantica_segmentador")
    eval4_results_dir = os.path.join(results_dir, "4_eval_activaciones_segmentador")

    os.makedirs(eval1_results_dir, exist_ok=True)
    os.makedirs(eval2_results_dir, exist_ok=True)
    os.makedirs(eval3_results_dir, exist_ok=True)
    os.makedirs(eval4_results_dir, exist_ok=True)

    print("\n" + "-" * 50)
    print("INICIANDO EVALUACIONES DE INTERPOLACIÓN")
    print("-" * 50)

    eval1_ssim, eval1_mse, eval1_l1 = run_eval_1(
        model=model,
        full_dataset=full_dataset,
        test_scan_to_indices=test_scan_to_indices,
        device=device,
        base_results_dir=eval1_results_dir,
        test_batch_size=test_batch_size,
        shared_example_paths=shared_example_paths,
        save_every_n_batches=save_every_n_batches,
    )

    all_metrics.append(
        {
            "Eval Type": "1. Síntesis Interp. vs Img GT Medio",
            "SSIM": eval1_ssim,
            "MSE": eval1_mse,
            "L1": eval1_l1,
        }
    )

    eval2_pixel_acc, eval2_dice, eval2_iou = run_eval_2(
        full_dataset=full_dataset,
        test_scan_to_indices=test_scan_to_indices,
        device=device,
        base_results_dir=eval2_results_dir,
        test_batch_size=test_batch_size,
        shared_example_paths=shared_example_paths,
        save_every_n_batches=save_every_n_batches,
    )

    semantic_metrics.append(
        {
            "Eval Type": "2. Interp. Máscara vs Mask GT Medio",
            "Pixel Acc": eval2_pixel_acc,
            "Dice": eval2_dice,
            "IoU": eval2_iou,
        }
    )

    print("\n" + "-" * 50)
    print("INICIANDO EVALUACIÓN SEMÁNTICA CON SEGMENTADOR")
    print("-" * 50)

    eval3_pixel_acc, eval3_dice, eval3_iou, eval3_global_metrics = run_eval_3(
        model=model,
        full_dataset=full_dataset,
        test_indices=test_indices,
        device=device,
        base_results_dir=eval3_results_dir,
        test_batch_size=test_batch_size,
        segmentador_model=None,
        shared_example_paths=shared_example_paths,
        save_every_n_batches=save_every_n_batches,
        num_classes=6,
        use_tta=use_tta,            
        tta_fn=partial(apply_geometric_tta, base_seed=tta_seed),
        elastic=elastic,
    )

    semantic_metrics.append(
        {
            "Eval Type": "3. Segmentador (Real vs Sintética)",
            "Pixel Acc": eval3_pixel_acc,
            "Dice": eval3_dice,
            "IoU": eval3_iou,
        }
    )

    print("\n" + "-" * 50)
    print("INICIANDO EVALUACIÓN DE ACTIVACIONES INTERMEDIAS")
    print("-" * 50)

    eval4_layer_metrics = run_eval_4(
        model=model,
        full_dataset=full_dataset,
        test_indices=test_indices,
        device=device,
        base_results_dir=eval4_results_dir,
        test_batch_size=test_batch_size,
        segmentador_model=None,
        target_layers={
            "decoder1": "decoder1",
            "decoder2": "decoder2",
            "decoder3": "decoder3",
        },
        num_classes=6,
    )

    activation_metrics.extend(eval4_layer_metrics)

    reconstruction_table = build_reconstruction_metrics_table(all_metrics)
    semantic_table = build_semantic_metrics_table(semantic_metrics)
    activation_table = build_activation_metrics_table(activation_metrics)

    final_report = (
        reconstruction_table
        + "\n\n"
        + semantic_table
        + "\n\n"
        + activation_table
        + "\n"
    )

    print("\n" + "=" * 75)
    print(final_report)
    print("=" * 75 + "\n")

    metrics_path = os.path.join(results_dir, "metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    latex_report = build_latex_metrics_report(
        all_metrics=all_metrics,
        semantic_metrics=semantic_metrics,
        activation_metrics=activation_metrics,
    )

    metrics_tex_path = os.path.join(results_dir, "metrics_tex.txt")

    with open(metrics_tex_path, "w", encoding="utf-8") as f:
        f.write(latex_report)

    print(f"Tabla de métricas guardada en: {metrics_path}")
    print(f"Tabla en formato LaTeX guardada en: {metrics_tex_path}")
    print(f"Las imágenes están organizadas en las subcarpetas dentro de: {results_dir}")


def save_visual_comparison(cond, target, output, idx, folder, img_path="unknown"):
    mask_vis = cond.cpu().numpy().argmax(0)
    real_vis = target[0].cpu().numpy()
    pred_vis = output[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(mask_vis, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0].set_title("Máscara (Entrada)")
    axes[0].axis("off")

    axes[1].imshow(real_vis, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("OCT Real (Ground Truth)")
    axes[1].axis("off")

    axes[2].imshow(pred_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("OCT Generado (Predicción)")
    axes[2].axis("off")

    fig.suptitle(f"Imagen: {img_path}", fontsize=10, y=0.98)

    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"std_comparativa_{idx}.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de evaluación completa del modelo")
    parser.add_argument("checkpoint", type=str, help="Ruta al checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size para test")
    parser.add_argument("--save_every", type=int, default=10, help="Guardar comparativas cada N batches")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "OCT_5k"),
        help="Ruta al dataset OCT_5k",
    )
    # --- NUEVO FLAG PARA TTA ---
    parser.add_argument("--tta", action="store_true", default=False, help="Activar Test-Time Augmentation (solo transformaciones geométricas)")
    parser.add_argument(
        "--tta-seed",
        type=int,
        default=12345,
        help="Semilla base para aplicar siempre la misma TTA a cada imagen",
    )

    args = parser.parse_args()

    run_test(
        checkpoint_path=args.checkpoint,
        test_batch_size=args.batch_size,
        save_every_n_batches=args.save_every,
        data_root=args.data_root,
        use_tta=args.tta,
        tta_seed=args.tta_seed,
    )