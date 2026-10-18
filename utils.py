import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_scan_id(sample):
    path = Path(sample["image"])
    return path.parent.name

def plot_loss_curves(train_losses, val_losses=None, log_dir="runs", experiment_name=""):
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _smooth_curve(values, window_size=10):
        if values is None:
            return None
        if len(values) < 2 or window_size <= 1:
            return list(values)

        window_size = min(window_size, len(values))
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed = np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")
        prefix = [float(values[0])] * (window_size - 1)
        return prefix + smoothed.tolist()

    def _save_curve(train_data, val_data, epochs_data, out_name, title_suffix=""):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs_data, train_data, label='Train Loss', linewidth=2)
        if val_data and len(val_data) > 0:
            ax.plot(epochs_data, val_data, label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_yscale('log')
        title = f'Loss Curves - {experiment_name}'
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, out_name), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _save_smoothed_curve(train_data, val_data, epochs_data, out_name, title_suffix="", window_size=10):
        smooth_train = _smooth_curve(train_data, window_size=window_size)
        smooth_val = _smooth_curve(val_data, window_size=window_size) if val_data and len(val_data) > 0 else None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epochs_data, train_data, label='Train Loss', linewidth=1.2, alpha=0.35)
        ax.plot(epochs_data, smooth_train, label=f'Train Loss (smoothed, w={window_size})', linewidth=2.5)
        if val_data and len(val_data) > 0:
            ax.plot(epochs_data, val_data, label='Val Loss', linewidth=1.2, alpha=0.35)
            ax.plot(epochs_data, smooth_val, label=f'Val Loss (smoothed, w={window_size})', linewidth=2.5)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_yscale('log')
        title = f'Loss Curves - {experiment_name}'
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, out_name), dpi=150, bbox_inches='tight')
        plt.close(fig)

    total_epochs = len(train_losses)
    full_epochs = list(range(1, total_epochs + 1))
    _save_curve(train_losses, val_losses, full_epochs, "loss_curves.png")
    _save_smoothed_curve(train_losses, val_losses, full_epochs, "loss_curves_smoothed.png")

    last_100 = min(100, total_epochs)
    start_100 = total_epochs - last_100
    epochs_100 = list(range(start_100 + 1, total_epochs + 1))
    val_100 = val_losses[start_100:] if val_losses and len(val_losses) > 0 else None
    _save_curve(
        train_losses[start_100:],
        val_100,
        epochs_100,
        "loss_curves_last100.png",
        f"Last {last_100} Epochs"
    )
    _save_smoothed_curve(
        train_losses[start_100:],
        val_100,
        epochs_100,
        "loss_curves_last100_smoothed.png",
        f"Last {last_100} Epochs",
    )

    last_25 = min(25, total_epochs)
    start_25 = total_epochs - last_25
    epochs_25 = list(range(start_25 + 1, total_epochs + 1))
    val_25 = val_losses[start_25:] if val_losses and len(val_losses) > 0 else None
    _save_curve(
        train_losses[start_25:],
        val_25,
        epochs_25,
        "loss_curves_last25.png",
        f"Last {last_25} Epochs"
    )
    _save_smoothed_curve(
        train_losses[start_25:],
        val_25,
        epochs_25,
        "loss_curves_last25_smoothed.png",
        f"Last {last_25} Epochs",
    )