import argparse
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import OCT5KDataset
from semantic_metrics import (
    compute_confusion_matrix,
    compute_metrics_from_confusion,
)
from segmentador_utils import (
    DEFAULT_EVAL_CLASSES,
    DEFAULT_GT_IGNORED_CLASS,
    load_segmentador,
    predict_segmentador_mask_remapped,
    prepare_gt_for_segmentador_display,
    get_valid_pixels_without_class,
    mask_to_palette,
)


def get_scan_id(sample):
    path = Path(sample["image"])
    return path.parent.name


def build_test_split(full_dataset, train_fraction=0.8):
    """
    Reproduce el split usado en test.py:
        - agrupa por scan
        - usa el 80% inicial para train/val
        - usa el 20% final para test
    """
    scan_to_indices = defaultdict(list)

    for idx, sample in enumerate(full_dataset.samples):
        scan_to_indices[get_scan_id(sample)].append(idx)

    scan_ids = sorted(list(scan_to_indices.keys()))
    n_trainval = int(len(scan_ids) * train_fraction)

    test_scans = scan_ids[n_trainval:]
    test_indices = [idx for sid in test_scans for idx in scan_to_indices[sid]]

    return test_scans, test_indices


def build_shared_example_paths(full_dataset, test_indices, test_batch_size, every_n_batches=10):
    selected_paths = set()

    for batch_idx, subset_start in enumerate(range(0, len(test_indices), test_batch_size)):
        if every_n_batches is None or every_n_batches <= 0:
            continue

        if batch_idx % every_n_batches != 0:
            continue

        dataset_idx = test_indices[subset_start]
        selected_paths.add(str(full_dataset.samples[dataset_idx]["image"]))

    return selected_paths


def add_metric_metadata(
    metrics,
    num_classes,
    ignored_class,
    eval_classes,
    total_pixels=0,
    ignored_pixels=0,
):
    """
    Añade campos de información global a las métricas.
    """
    metrics["num_classes"] = num_classes
    metrics["ignored_class"] = ignored_class
    metrics["eval_classes"] = eval_classes
    metrics["total_pixels"] = int(total_pixels)
    metrics["ignored_pixels"] = int(ignored_pixels)

    if total_pixels > 0:
        metrics["ignored_percent"] = 100.0 * ignored_pixels / total_pixels
    else:
        metrics["ignored_percent"] = float("nan")

    return metrics


def build_eval3_summary_table(global_metrics):
    """
    Construye una tabla compacta para metrics.txt.
    """
    row_name = "3. Segmentador (Real vs Sintética)"

    header = (
        f"| {'Tipo de Evaluación':<42} | "
        f"{'Pixel Acc':<9} | "
        f"{'Dice':<8} | "
        f"{'IoU':<8} |"
    )

    separator = (
        "|"
        + "-" * 44
        + "|"
        + "-" * 11
        + "|"
        + "-" * 10
        + "|"
        + "-" * 10
        + "|"
    )

    line = (
        f"| {row_name:<42} | "
        f"{global_metrics['pixel_accuracy']:<9.4f} | "
        f"{global_metrics['mean_dice']:<8.4f} | "
        f"{global_metrics['mean_iou']:<8.4f} |"
    )

    table = "\n".join(
        [
            "RESUMEN DE MÉTRICAS SEMÁNTICAS",
            "==============================",
            "",
            header,
            separator,
            line,
            "",
        ]
    )

    return table


def save_semantic_comparison_figure(
    real_img,
    synth_img,
    seg_real,
    seg_synth,
    gt_no_lower,
    valid_mask,
    idx,
    folder,
    img_path="unknown",
):
    """
    Guarda una figura con:
        1. OCT real
        2. OCT sintética
        3. Segmentación de OCT real
        4. Segmentación de OCT sintética
        5. GT sin capa inferior
        6. Mapa de aciertos/errores semánticos desglosado
    """
    os.makedirs(folder, exist_ok=True)

    real_np = real_img.detach().cpu().numpy()[0]
    synth_np = synth_img.detach().cpu().numpy()[0]

    seg_real_cpu = seg_real.detach().cpu()
    seg_synth_cpu = seg_synth.detach().cpu()
    valid_cpu = valid_mask.detach().cpu().bool()

    seg_real_rgb = mask_to_palette(seg_real_cpu)
    seg_synth_rgb = mask_to_palette(seg_synth_cpu)
    gt_no_lower_rgb = mask_to_palette(gt_no_lower)

    # --- NUEVO MAPA DE ERROR DESGLOSADO ---
    error_map = torch.zeros_like(seg_real_cpu, dtype=torch.long)

    # 1. Aciertos (Gris)
    correct = (seg_real_cpu == seg_synth_cpu) & valid_cpu
    
    # 2. Falsos Positivos: El segmentador vio tejido (>0) donde en realidad había fondo (0)
    fp = (seg_synth_cpu > 0) & (seg_real_cpu == 0) & valid_cpu
    
    # 3. Falsos Negativos: El segmentador vio fondo (0) donde en realidad había tejido (>0)
    fn = (seg_synth_cpu == 0) & (seg_real_cpu > 0) & valid_cpu
    
    # 4. Confusión de capa: Vio tejido, pero se equivocó de clase
    confusion = (seg_synth_cpu > 0) & (seg_real_cpu > 0) & (seg_real_cpu != seg_synth_cpu) & valid_cpu

    # Asignar los valores al mapa
    error_map[correct] = 1
    error_map[fp] = 2
    error_map[fn] = 3
    error_map[confusion] = 4

    # Crear el nuevo mapa de colores: 0=Negro, 1=Gris, 2=Rojo, 3=Azul, 4=Amarillo
    error_cmap = mcolors.ListedColormap(["black", "lightgray", "red", "blue", "yellow"])
    error_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], error_cmap.N)
    # --------------------------------------

    fig, axes = plt.subplots(1, 6, figsize=(34, 6))

    axes[0].imshow(real_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("OCT real")
    axes[0].axis("off")

    axes[1].imshow(synth_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("OCT sintética")
    axes[1].axis("off")

    axes[2].imshow(seg_real_rgb)
    axes[2].set_title("Segmentación real")
    axes[2].axis("off")

    axes[3].imshow(seg_synth_rgb)
    axes[3].set_title("Segmentación sintética")
    axes[3].axis("off")

    axes[4].imshow(gt_no_lower_rgb)
    axes[4].set_title("GT sin capa inferior")
    axes[4].axis("off")

    axes[5].imshow(
        error_map.numpy(),
        cmap=error_cmap,
        norm=error_norm,
        interpolation="nearest",
    )
    axes[5].set_title("Error semántico\ngris=ok, rojo=FP, azul=FN, amar=conf")
    axes[5].axis("off")

    fig.suptitle(f"Imagen: {img_path}", fontsize=10, y=0.98)
    plt.tight_layout()

    output_path = os.path.join(folder, f"eval3_semantic_{idx}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_eval_3(
    model,
    full_dataset,
    test_indices,
    device,
    base_results_dir,
    test_batch_size=4,
    segmentador_model=None,
    shared_example_paths=None,
    save_every_n_batches=10,
    num_classes=6,
    ignored_class=DEFAULT_GT_IGNORED_CLASS,
    eval_classes=DEFAULT_EVAL_CLASSES,
    use_tta=False,
    tta_fn=None,
    elastic=None,
):
    """
    Eval 3: evaluación semántica mediante segmentador.

    Compara:
        segmentador(imagen real)
        vs
        segmentador(imagen sintética)

    La imagen sintética se genera como:
        máscara GT -> one-hot -> sintetizador -> OCT sintética
    """
    print("\n--- Iniciando Eval 3: Semántica mediante segmentador ---")

    if shared_example_paths is None:
        shared_example_paths = set()

    results_dir = base_results_dir
    os.makedirs(results_dir, exist_ok=True)

    if segmentador_model is None:
        print("Cargando segmentador externo...")
        segmentador_model = load_segmentador(device=device)

    test_loader = DataLoader(
        Subset(full_dataset, test_indices),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model.eval()

    total_confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    total_pixels = 0
    total_ignored_pixels = 0

    pbar = tqdm(test_loader, desc="Eval 3 Semantic")

    for i, batch in enumerate(pbar):
        real_img = batch["image"].to(device).float()
        raw_mask = batch["mask"].to(device).long()

        # --- APLICAR TTA SI ESTÁ HABILITADO ---
        if use_tta and tta_fn is not None and elastic is not None:
            real_img, raw_mask = tta_fn(real_img, raw_mask, elastic)
            
            # Limpiar la clase 10 (bordes negros introducidos por TTA) 
            # convirtiéndola en clase 0 (fondo) para que el one_hot no falle
            valid_tta_mask = (raw_mask != 10)
            raw_mask = torch.where(raw_mask == 10, 0, raw_mask)
        # --------------------------------------

        # Asegurar que el one_hot nunca pida una clase fuera de rango (clamp de 0 a 5)
        cond = OCT5KDataset.mask_to_one_hot(
            raw_mask.clamp(0, 5),
            num_classes=num_classes,
        ).float()

        raw_output = model(cond)
        synth_img = torch.sigmoid(raw_output.float())

        seg_real = predict_segmentador_mask_remapped(
            segmentador_model=segmentador_model,
            image_tensor=real_img,
            device=device,
        )

        seg_synth = predict_segmentador_mask_remapped(
            segmentador_model=segmentador_model,
            image_tensor=synth_img,
            device=device,
        )

        valid_mask = get_valid_pixels_without_class(
            raw_mask,
            class_id_to_ignore=ignored_class,
        )
        
        # Ignorar los bordes introducidos por TTA en la evaluación semántica
        if use_tta and tta_fn is not None and elastic is not None:
            valid_mask = valid_mask & valid_tta_mask

        batch_confusion = compute_confusion_matrix(
            pred_mask=seg_synth.detach().cpu(),
            gt_mask=seg_real.detach().cpu(),
            num_classes=num_classes,
            valid_mask=valid_mask.detach().cpu(),
        )

        total_confusion += batch_confusion

        batch_total_pixels = int(raw_mask.numel())
        batch_ignored_pixels = int((raw_mask == ignored_class).sum().item())

        total_pixels += batch_total_pixels
        total_ignored_pixels += batch_ignored_pixels

        batch_metrics = compute_metrics_from_confusion(
            confusion_matrix=batch_confusion,
            eval_classes=eval_classes,
        )

        pbar.set_postfix(
            {
                "Dice": f"{batch_metrics['mean_dice']:.4f}",
                "IoU": f"{batch_metrics['mean_iou']:.4f}",
            }
        )

        img_path = (
            batch["meta"]["image_path"][0]
            if "meta" in batch and "image_path" in batch["meta"]
            else "unknown"
        )

        should_save = (
            save_every_n_batches is not None
            and save_every_n_batches > 0
            and i % save_every_n_batches == 0
        )

        if img_path in shared_example_paths:
            should_save = True

        if should_save:
            gt_raw_cpu = raw_mask[0].detach().cpu()
            gt_no_lower = prepare_gt_for_segmentador_display(
                gt_raw_cpu,
                ignored_class=ignored_class,
                fill_value=0,
            )

            save_semantic_comparison_figure(
                real_img=real_img[0].detach().cpu(),
                synth_img=synth_img[0].detach().cpu(),
                seg_real=seg_real[0].detach().cpu(),
                seg_synth=seg_synth[0].detach().cpu(),
                gt_no_lower=gt_no_lower,
                valid_mask=valid_mask[0].detach().cpu(),
                idx=i,
                folder=results_dir,
                img_path=img_path,
            )

    global_metrics = compute_metrics_from_confusion(
        confusion_matrix=total_confusion,
        eval_classes=eval_classes,
    )

    global_metrics = add_metric_metadata(
        metrics=global_metrics,
        num_classes=num_classes,
        ignored_class=ignored_class,
        eval_classes=eval_classes,
        total_pixels=total_pixels,
        ignored_pixels=total_ignored_pixels,
    )

    print("\n" + "=" * 80)
    print("EVAL 3 - RESUMEN GLOBAL")
    print("=" * 80)
    print(
        f"Semantic Pixel Acc: {global_metrics['pixel_accuracy']:.6f} | "
        f"Semantic Dice: {global_metrics['mean_dice']:.6f} | "
        f"Semantic IoU: {global_metrics['mean_iou']:.6f}"
    )

    for cls, values in global_metrics["per_class"].items():
        print(
            f"  Clase {cls}: "
            f"Dice={values['dice']:.6f}, "
            f"IoU={values['iou']:.6f}, "
            f"RealSeg pixels={int(values['support_gt'])}, "
            f"SynthSeg pixels={int(values['support_pred'])}"
        )

    print(f"\nEval 3 finalizada.")
    print(f"Figuras guardadas en: {results_dir}")

    return (
        global_metrics["pixel_accuracy"],
        global_metrics["mean_dice"],
        global_metrics["mean_iou"],
        global_metrics,
    )