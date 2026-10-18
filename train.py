import os
import argparse
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torchvision import tv_tensors
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend sin GUI, solo para guardar ficheros
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import segmentation_models_pytorch as smp
import albumentations as A
import cv2

# Imports de módulos propios
from dataset import OCT5KDataset
from utils import get_scan_id, plot_loss_curves
from config_manager import load_config_interactively, load_config_from_cli, load_checkpoint
from losses import get_criterion

torch.set_float32_matmul_precision('high')

MASK_COLORS = ["#1f77b4", "#17becf", "#8c564b", "#7f7f7f", "#2ca02c", "#d62728"]
MASK_CMAP = mcolors.ListedColormap(MASK_COLORS)
MASK_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], MASK_CMAP.N)


def build_transform(use_augmentation: bool):
    if not use_augmentation:
        return None
    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0, fill=0, fill_mask=10),
        A.HorizontalFlip(p=0.5),
        #A.RandomResizedCrop(size=(512, 512), scale=(0.90, 1.0)),
    ])

def apply_augmentation_gpu(img: torch.Tensor, mask: torch.Tensor,
                            elastic: v2.ElasticTransform,
                            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:

    b, _, h, w = img.shape

    # Elastic transform (p=0.5)
    
    if torch.rand(1).item() < 0.5:
        img_tv  = tv_tensors.Image(img)
        mask_tv = tv_tensors.Mask(mask.unsqueeze(1))
        img_tv, mask_tv = elastic(img_tv, mask_tv)
        img  = img_tv.as_subclass(torch.Tensor)
        mask = mask_tv.as_subclass(torch.Tensor).squeeze(1)
        

    # Gamma: simula variaciones en sensibilidad del detector
    if torch.rand(1).item() < 0.5:
        gamma = torch.empty(1).uniform_(0.75, 1.5).item()
        img   = img.clamp(0.0, 1.0).pow(gamma)

    # Gaussian blur: suaviza artefactos
    if torch.rand(1).item() < 0.8:
        sigma = torch.empty(1).uniform_(0.5, 2.0).item()
        ks    = max(int(sigma * 6) | 1, 3)
        img   = v2.functional.gaussian_blur(img, kernel_size=[ks, ks], sigma=[sigma, sigma])

    return img, mask


def save_training_batch_preview(batch, original_dataset, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    images = batch["image"].detach().cpu()
    masks = batch["mask"].detach().cpu()
    image_paths = batch["meta"]["image_path"]

    batch_size = min(images.shape[0], 4)
    fig, axes = plt.subplots(batch_size, 4, figsize=(18, 4 * batch_size))
    if batch_size == 1:
        axes = np.array([axes])

    for i in range(batch_size):
        image_path = image_paths[i]
        original_idx = next(
            idx for idx, sample in enumerate(original_dataset.samples)
            if sample["image_path_str"] == image_path
        )
        original_sample = original_dataset[original_idx]

        # Columna 0: imagen original sin augmentation
        axes[i, 0].imshow(original_sample["image"].squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Original {i}")
        axes[i, 0].axis("off")

        # Columna 1: imagen augmentada (Albumentations CPU)
        axes[i, 1].imshow(images[i].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Augmented CPU {i}")
        axes[i, 1].axis("off")

        # Columna 2: máscara original
        axes[i, 2].imshow(
            original_sample["mask"].cpu().numpy(),
            cmap=MASK_CMAP,
            norm=MASK_NORM,
            interpolation="nearest",
        )
        axes[i, 2].set_title(f"Mask original {i}")
        axes[i, 2].axis("off")

        # Columna 3: máscara augmentada
        axes[i, 3].imshow(
            masks[i].numpy(),
            cmap=MASK_CMAP,
            norm=MASK_NORM,
            interpolation="nearest",
        )
        axes[i, 3].set_title(f"Mask augmented {i}")
        axes[i, 3].axis("off")

    fig.suptitle("Primer batch: original vs augmentada (CPU)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo OCT")
    parser.add_argument("--name",        type=str,   default=None)
    parser.add_argument("--loss_type",   type=str,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--resume_ckpt", type=str,   default=None)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "OCT_5k"),
        help="Ruta al directorio OCT_5k.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de procesos del DataLoader.",
    )
    parser.add_argument("--augment",     action="store_true", default=False)
    parser.add_argument("--preview_aug_gpu", action="store_true", default=False)
    parser.add_argument("--preview_aug_multi", type=int, default=0) 


    args = parser.parse_args()

    if args.name and args.loss_type and args.lr and args.batch_size and args.epochs:
        config_data = load_config_from_cli(args)
    else:
        config_data = load_config_interactively()
        if not config_data:
            return

    # Semilla
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ejecutando en: {device}")

    use_augmentation = args.augment
    print(f"Data augmentation: {'ACTIVADA' if use_augmentation else 'desactivada'}")

    aug_tag = "_aug" if use_augmentation else ""
    log_name = (
        f"{config_data['name']}_loss{config_data['loss_type']}"
        f"_lr{config_data['lr']}_bs{config_data['batch_size']}"
        f"_ep{config_data['epochs']}{aug_tag}"
    )

    # Dataset: el transform se aplica dentro de __getitem__
    full_dataset = OCT5KDataset(
        data_root=args.data_root,
        load_segmentation=True,
        return_pair=True,
        transform=build_transform(use_augmentation)
    )
    original_dataset = OCT5KDataset(
        data_root=args.data_root,
        load_segmentation=True,
        return_pair=True,
        transform=None,
        preload=False,
    )

    scan_to_indices = defaultdict(list)
    for idx, sample in enumerate(full_dataset.samples):
        scan_to_indices[get_scan_id(sample)].append(idx)

    scan_ids      = sorted(list(scan_to_indices.keys()))
    n_train       = int(len(scan_ids) * 0.8)
    train_indices = [idx for sid in scan_ids[:n_train] for idx in scan_to_indices[sid]]

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=config_data['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if args.preview_aug_multi > 0:
        elastic_preview = v2.ElasticTransform(alpha=[500, 5000], sigma=[50, 50], fill=10)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.preview_aug_multi:
                break

            img  = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            if use_augmentation:
                img, mask = apply_augmentation_gpu(img, mask, elastic_preview, device)

                b, _, h, w = img.shape
                scale = torch.empty(1).uniform_(0.80, 1.0).item()
                crop_h = int(h * scale)
                crop_w = int(w * scale)
                top  = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                img  = img[:, :, top:top+crop_h, left:left+crop_w]
                mask = mask[:, top:top+crop_h, left:left+crop_w]
                img  = torch.nn.functional.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()

                sigma_speckle = torch.empty(1).uniform_(1.0 / 255.0, 4.0 / 255.0).item()
                noise = torch.randn_like(img) * sigma_speckle
                img = (img + img * noise).clamp(0.0, 1.0)

            n = min(4, img.shape[0])
            image_paths = batch["meta"]["image_path"]
            preview_path = os.path.join("runs", "aug_gpu_preview_multi", f"batch_{batch_idx}.png")
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)

            fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
            if n == 1:
                axes = np.array([axes])

            for i in range(n):
                original_idx = next(
                    idx for idx, sample in enumerate(original_dataset.samples)
                    if sample["image_path_str"] == image_paths[i]
                )
                original_sample = original_dataset[original_idx]

                axes[i, 0].imshow(original_sample["image"].squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                axes[i, 0].set_title(f"Original {i}")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(img[i].squeeze(0).cpu().float().numpy(), cmap="gray", vmin=0, vmax=1)
                axes[i, 1].set_title(f"DA {i}")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(
                    original_sample["mask"].cpu().numpy(),
                    cmap=MASK_CMAP,
                    norm=MASK_NORM,
                    interpolation="nearest",
                )
                axes[i, 2].set_title(f"Mask original {i}")
                axes[i, 2].axis("off")

                axes[i, 3].imshow(
                    mask[i].cpu().numpy(),
                    cmap=MASK_CMAP,
                    norm=MASK_NORM,
                    interpolation="nearest",
                )
                axes[i, 3].set_title(f"Mask DA {i}")
                axes[i, 3].axis("off")

            plt.tight_layout()
            plt.savefig(preview_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Preview batch {batch_idx} guardado en: {preview_path}")

        return  # salir sin entrenar

    model     = smp.Unet(encoder_name="resnet34", in_channels=6, classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config_data['lr'], weight_decay=0.05)
    criterion = get_criterion(config_data['loss_type'])

    start_epoch, global_step = 0, 0
    train_losses = []
    ckpt_path = args.resume if args.resume else config_data.get("resume_ckpt")
    if ckpt_path:
        try:
            start_epoch, global_step, train_losses = load_checkpoint(
                ckpt_path, model, optimizer, device, config_data
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    epochs = config_data['epochs']

    elastic = v2.ElasticTransform(alpha=[500, 5000], sigma=[50, 50], fill=10)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss       = 0
        epoch_start_time = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            img  = batch["image"].to(device, non_blocking=True)  # (B, 1, H, W)
            mask = batch["mask"].to(device, non_blocking=True)   # (B, H, W)

            # Aplicar elastic transform sobre el batch completo
            if use_augmentation:
                img, mask = apply_augmentation_gpu(img, mask, elastic, device)

                sigma_speckle = torch.empty(1).uniform_(0.5/ 255.0, 2.0 / 255.0).item()
                noise = torch.randn_like(img) * sigma_speckle
                img = (img + img * noise).clamp(0.0, 1.0)

                # RandomResizedCrop al final de todo
                b, _, h, w = img.shape
                scale = torch.empty(1).uniform_(0.80, 1.0).item()
                crop_h = int(h * scale)
                crop_w = int(w * scale)
                top  = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                img  = img[:, :, top:top+crop_h, left:left+crop_w]
                mask = mask[:, top:top+crop_h, left:left+crop_w]
                img  = torch.nn.functional.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()



            if args.preview_aug_gpu and epoch == start_epoch:
                n = min(4, img.shape[0])
                image_paths = batch["meta"]["image_path"]
                preview_path = os.path.join("runs", "aug_gpu_preview", "first_batch_augmented.png")
                os.makedirs(os.path.dirname(preview_path), exist_ok=True) 
                
                fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
                if n == 1:
                    axes = np.array([axes])
                
                for i in range(n):
                    # Recuperar original sin ninguna augmentation
                    original_idx = next(
                        idx for idx, sample in enumerate(original_dataset.samples)
                        if sample["image_path_str"] == image_paths[i]
                    )
                    original_sample = original_dataset[original_idx]
                    
                    axes[i, 0].imshow(original_sample["image"].squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, 0].set_title(f"Original {i}")
                    axes[i, 0].axis("off")

                    axes[i, 1].imshow(img[i].squeeze(0).cpu().float().numpy(), cmap="gray", vmin=0, vmax=1)
                    axes[i, 1].set_title(f"Augmented {i}")
                    axes[i, 1].axis("off")
                    
                    axes[i, 2].imshow(mask[i].cpu().numpy(), cmap="tab20", interpolation="nearest")
                    axes[i, 2].set_title(f"Mask {i}")
                    axes[i, 2].axis("off")
                
                plt.tight_layout()
                plt.savefig(preview_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Preview guardado en: {preview_path}")
                break


            valid = (mask != 10).float()  # (B, H, W)

            target = img.float()

            cond   = OCT5KDataset.mask_to_one_hot(mask.clamp(0, 5), num_classes=6).float()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                raw_output = model(cond)

            output = torch.sigmoid(raw_output.float())
            loss = criterion(output * valid.unsqueeze(1), target * valid.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        epoch_time = time.perf_counter() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{epochs} | train={avg_train_loss:.6f} | "
            f"time={epoch_time:.2f}s | aug={'on' if use_augmentation else 'off'}"
        )

        if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_name = (
                f"ckpt_{config_data['name']}_bs{config_data['batch_size']}"
                f"_ep{epoch+1}{aug_tag}.pth.tar"
            )
            torch.save(
                {
                    'epoch':        epoch + 1,
                    'global_step':  global_step,
                    'state_dict':   model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'train_losses': train_losses,
                    'config':       config_data,
                    'augmentation': use_augmentation,
                },
                os.path.join("checkpoints", ckpt_name),
            )

    plot_loss_curves(
        train_losses,
        log_dir=os.path.join("runs", log_name),
        experiment_name=config_data['name'],
    )


if __name__ == "__main__":
    train()