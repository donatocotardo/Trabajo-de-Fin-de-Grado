import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import OCT5KDataset
from segmentador_utils import load_segmentador


def get_scan_id(sample):
    path = Path(sample["image"])
    return path.parent.name


def build_test_split(full_dataset, train_fraction=0.8):
    scan_to_indices = defaultdict(list)

    for idx, sample in enumerate(full_dataset.samples):
        scan_to_indices[get_scan_id(sample)].append(idx)

    scan_ids = sorted(list(scan_to_indices.keys()))
    n_trainval = int(len(scan_ids) * train_fraction)

    test_scans = scan_ids[n_trainval:]
    test_indices = [idx for sid in test_scans for idx in scan_to_indices[sid]]

    return test_scans, test_indices


class ActivationExtractor:
    """
    Captura activaciones internas del segmentador usando forward hooks.
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.handles = []

        self._register_hooks()

    def _register_hooks(self):
        module_dict = dict(self.model.named_modules())

        missing_layers = []

        for display_name, module_name in self.target_layers.items():
            if module_name not in module_dict:
                missing_layers.append(module_name)
                continue

            module = module_dict[module_name]

            def hook_fn(module, input, output, name=display_name):
                if isinstance(output, (tuple, list)):
                    output = output[0]

                self.activations[name] = output.detach()

            handle = module.register_forward_hook(hook_fn)
            self.handles.append(handle)

        if len(missing_layers) > 0:
            available = "\n".join([name for name, _ in self.model.named_modules()])
            raise ValueError(
                "No se encontraron algunas capas en el segmentador:\n"
                + "\n".join(missing_layers)
                + "\n\nCapas disponibles:\n"
                + available
            )

    def clear(self):
        self.activations = {}

    @torch.no_grad()
    def forward_and_capture(self, x):
        """
        Ejecuta el segmentador y devuelve las activaciones capturadas.

        x debe tener forma:
            (1, 1, H, W)

        Se usa predict() porque el forward del segmentador requiere h_number.
        """
        self.clear()

        if hasattr(self.model, "predict"):
            _ = self.model.predict(x)
        else:
            try:
                _ = self.model(x)
            except TypeError:
                h_number = x.shape[-2]
                _ = self.model(x, h_number)

        missing = [name for name in self.target_layers.keys() if name not in self.activations]

        if len(missing) > 0:
            raise RuntimeError(
                "No se capturaron activaciones para las capas: "
                + ", ".join(missing)
            )

        return {k: v.detach() for k, v in self.activations.items()}

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles = []


def compute_feature_metrics(real_feat, synth_feat):
    """
    Compara dos tensores de activación.

    Métricas:
        - Feature MSE
        - Feature L1
        - Cosine similarity
    """
    if real_feat.shape != synth_feat.shape:
        raise ValueError(
            f"Las activaciones no tienen la misma forma: "
            f"real={tuple(real_feat.shape)}, synth={tuple(synth_feat.shape)}"
        )

    real_feat = real_feat.float()
    synth_feat = synth_feat.float()

    mse = F.mse_loss(synth_feat, real_feat, reduction="mean").item()
    l1 = F.l1_loss(synth_feat, real_feat, reduction="mean").item()

    real_flat = real_feat.flatten(start_dim=1)
    synth_flat = synth_feat.flatten(start_dim=1)

    cosine = F.cosine_similarity(
        synth_flat,
        real_flat,
        dim=1,
        eps=1e-8,
    ).mean().item()

    return mse, l1, cosine


def build_eval4_summary_table(layer_metrics):
    header = (
        f"| {'Tipo de Evaluación':<36} | "
        f"{'Layer':<10} | "
        f"{'Feature MSE':<12} | "
        f"{'Feature L1':<11} | "
        f"{'Cosine':<8} |"
    )

    separator = (
        "|"
        + "-" * 38
        + "|"
        + "-" * 12
        + "|"
        + "-" * 14
        + "|"
        + "-" * 13
        + "|"
        + "-" * 10
        + "|"
    )

    lines = [
        "RESUMEN DE ACTIVACIONES INTERMEDIAS",
        "===================================",
        "",
        header,
        separator,
    ]

    eval_name = "4. Activaciones Real vs Sintética"

    for row in layer_metrics:
        line = (
            f"| {eval_name:<36} | "
            f"{row['layer']:<10} | "
            f"{row['feature_mse']:<12.6f} | "
            f"{row['feature_l1']:<11.6f} | "
            f"{row['cosine']:<8.4f} |"
        )
        lines.append(line)

    lines.append("")

    return "\n".join(lines)


@torch.no_grad()
def run_eval_4(
    model,
    full_dataset,
    test_indices,
    device,
    base_results_dir,
    test_batch_size=4,
    segmentador_model=None,
    target_layers=None,
    num_classes=6,
):
    """
    Eval 4: comparación de activaciones internas del segmentador.

    Flujo:
        máscara GT -> sintetizador -> imagen sintética

        imagen real      -> segmentador -> activaciones decoder1/2/3
        imagen sintética -> segmentador -> activaciones decoder1/2/3

    Métricas por capa:
        - Feature MSE
        - Feature L1
        - Cosine similarity
    """
    print("\n--- Iniciando Eval 4: Activaciones intermedias del segmentador ---")

    os.makedirs(base_results_dir, exist_ok=True)

    if target_layers is None:
        target_layers = {
            "decoder1": "decoder1",
            "decoder2": "decoder2",
            "decoder3": "decoder3",
        }

    if segmentador_model is None:
        print("Cargando segmentador externo...")
        segmentador_model = load_segmentador(device=device)

    model.eval()
    segmentador_model.eval()

    extractor = ActivationExtractor(
        model=segmentador_model,
        target_layers=target_layers,
    )

    test_loader = DataLoader(
        Subset(full_dataset, test_indices),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    accum = {}

    for layer_name in target_layers.keys():
        accum[layer_name] = {
            "mse_sum": 0.0,
            "l1_sum": 0.0,
            "cosine_sum": 0.0,
            "n_samples": 0,
        }

    try:
        pbar = tqdm(test_loader, desc="Eval 4 Activations")

        for i, batch in enumerate(pbar):
            real_img = batch["image"].to(device).float()
            raw_mask = batch["mask"].to(device).long()

            cond = OCT5KDataset.mask_to_one_hot(
                raw_mask,
                num_classes=num_classes,
            ).float()

            raw_output = model(cond)
            synth_img = torch.sigmoid(raw_output.float())

            batch_size = real_img.shape[0]
            batch_cosines = {layer_name: [] for layer_name in target_layers.keys()}

            for b in range(batch_size):
                real_img_b = real_img[b:b + 1]
                synth_img_b = synth_img[b:b + 1]

                real_acts = extractor.forward_and_capture(real_img_b)
                synth_acts = extractor.forward_and_capture(synth_img_b)

                for layer_name in target_layers.keys():
                    mse, l1, cosine = compute_feature_metrics(
                        real_feat=real_acts[layer_name],
                        synth_feat=synth_acts[layer_name],
                    )

                    accum[layer_name]["mse_sum"] += mse
                    accum[layer_name]["l1_sum"] += l1
                    accum[layer_name]["cosine_sum"] += cosine
                    accum[layer_name]["n_samples"] += 1

                    batch_cosines[layer_name].append(cosine)

            postfix = {}

            for layer_name in target_layers.keys():
                values = batch_cosines[layer_name]

                if len(values) > 0:
                    avg_cos = sum(values) / len(values)
                    postfix[f"{layer_name}_cos"] = f"{avg_cos:.3f}"

            pbar.set_postfix(postfix)

    finally:
        extractor.remove()

    layer_metrics = []

    print("\n" + "=" * 80)
    print("EVAL 4 - RESUMEN GLOBAL")
    print("=" * 80)

    for layer_name in target_layers.keys():
        n = accum[layer_name]["n_samples"]

        feature_mse = accum[layer_name]["mse_sum"] / max(n, 1)
        feature_l1 = accum[layer_name]["l1_sum"] / max(n, 1)
        cosine = accum[layer_name]["cosine_sum"] / max(n, 1)

        row = {
            "layer": layer_name,
            "feature_mse": feature_mse,
            "feature_l1": feature_l1,
            "cosine": cosine,
        }

        layer_metrics.append(row)

        print(
            f"{layer_name}: "
            f"Feature MSE={feature_mse:.6f} | "
            f"Feature L1={feature_l1:.6f} | "
            f"Cosine={cosine:.6f}"
        )

    print(f"\nEval 4 finalizada.")
    print(f"Carpeta de resultados: {base_results_dir}")

    return layer_metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Eval 4: comparación de activaciones internas del segmentador."
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Ruta al checkpoint del sintetizador.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "OCT_5k"),
        help="Ruta al dataset OCT_5k.",
    )

    parser.add_argument(
        "--grading",
        type=str,
        default="Grading_1",
        help="Grading usado para las máscaras.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size para evaluación.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_eval4_activations",
        help="Carpeta donde guardar resultados.",
    )

    return parser


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    data_root = Path(args.data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"No existe data-root: {data_root}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"No existe checkpoint: {args.checkpoint}")

    full_dataset = OCT5KDataset(
        data_root=str(data_root),
        grading=args.grading,
        load_segmentation=True,
        return_pair=True,
        preload=False,
        transform=None,
    )

    test_scans, test_indices = build_test_split(full_dataset)

    print(f"Dataset cargado: {len(full_dataset)} imágenes")
    print(f"Scans de test: {len(test_scans)}")
    print(f"Imágenes de test: {len(test_indices)}")

    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=6,
        classes=1,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Pesos del sintetizador cargados desde: {args.checkpoint}")

    segmentador_model = load_segmentador(device=device)

    run_eval_4(
        model=model,
        full_dataset=full_dataset,
        test_indices=test_indices,
        device=device,
        base_results_dir=args.output_dir,
        test_batch_size=args.batch_size,
        segmentador_model=segmentador_model,
        target_layers={
            "decoder1": "decoder1",
            "decoder2": "decoder2",
            "decoder3": "decoder3",
        },
        num_classes=6,
    )


if __name__ == "__main__":
    main()