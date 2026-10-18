import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision import tv_tensors

class OCT5KDataset(Dataset):

    def __init__(
        self,
        data_root,
        grading="Grading_1",
        crop_size=(512,512),
        load_segmentation=True,
        load_boundaries=False,
        load_detection=False,
        return_pair=False,
        preload=True,  
        transform=None 
    ):
        self.data_root = Path(data_root)
        self.grading = grading
        self.crop_size = crop_size

        self.load_segmentation = load_segmentation
        self.load_boundaries = load_boundaries
        self.load_detection = load_detection

        self.return_pair = return_pair
        self.preload = preload

        self.transform = transform

        # _index_dataset devuelve rutas
        self.samples = self._index_dataset()

        # Si preload=True, carga todo en RAM
        if self.preload:
            self._preload_data()
        
    def _index_dataset(self):
        samples = []

        images_root = self.data_root / "Images" / "Images_Manual"
        masks_root = self.data_root / "Masks" / "Masks_Manual" / self.grading
        boundaries_root = self.data_root / "Boundaries" / "Boundaries_Manual" / self.grading

        for pathology_dir in sorted(images_root.iterdir()):
            if not pathology_dir.is_dir():
                continue

            pathology = pathology_dir.name.split()[0]

            for root, _, files in os.walk(pathology_dir):
                for fname in sorted(files):
                    if not fname.endswith(".png"):
                        continue

                    img_path = Path(root) / fname
                    rel = img_path.relative_to(images_root / pathology_dir.name)

                    sample = {
                        "image": img_path,
                        "image_path_str": str(img_path),   # precalculado una vez
                        "pathology": pathology
                    }

                    if self.load_segmentation:
                        sample["mask"] = masks_root / pathology_dir.name / rel

                    if self.load_boundaries:
                        sample["boundaries"] = boundaries_root / pathology_dir.name / rel.with_suffix(".csv")

                    samples.append(sample)

        if len(samples) == 0:
            raise RuntimeError("No samples found. Check dataset paths.")

        return samples

    # Lee todos los ficheros, en el constructor.
    # Tras esto, __getitem__ nunca toca el disco.
    def _preload_data(self):

        print(f"Precargando {len(self.samples)} muestras en RAM...")
        self._imgs   = []   # lista de tensores (1,H,W) float32 en [0,1]
        self._masks  = []   # lista de tensores (H,W) int64, o None

        for s in tqdm(self.samples, desc="Preload"):
            # Imagen: escala de grises → float32 [0,1] → tensor (1,H,W)
            img = Image.open(s["image"]).convert("L")
            img_t = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).unsqueeze(0)  # (1,H,W)
            self._imgs.append(img_t)

            # Máscara
            if self.load_segmentation:
                mask = Image.open(s["mask"]).convert("L")
                mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
                self._masks.append(mask_t)
            else:
                self._masks.append(None)

        print("Precarga completada.")

    @staticmethod
    def mask_to_one_hot(mask: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
        # Creamos el tensor de clases directamente en el mismo dispositivo que la máscara (GPU)
        # Forma: (1, num_classes, 1, 1)
        classes = torch.arange(num_classes, device=mask.device, dtype=mask.dtype).view(1, -1, 1, 1)
        
        # mask.unsqueeze(1) le da la forma (B, 1, H, W)
        # Al compararlos, PyTorch hace el broadcast perfecto a (B, num_classes, H, W)
        return (classes == mask.unsqueeze(1)).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.preload:
            img  = self._imgs[idx]
            mask = self._masks[idx]   
        else:
            img_pil = Image.open(s["image"]).convert("L")
            img = torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255.0).unsqueeze(0)
            mask = None
            if self.load_segmentation:
                mask_pil = Image.open(s["mask"]).convert("L")
                mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))

       # Aplicar augmentation si existe
        if self.transform is not None:
            img_np   = (img.squeeze(0).numpy() * 255).astype(np.uint8) 
            mask_np  = mask.numpy().astype(np.uint8)                    

            result   = self.transform(image=img_np, mask=mask_np)

            img  = torch.from_numpy(result["image"].astype(np.float32) / 255.0).unsqueeze(0)
            mask = torch.from_numpy(result["mask"].astype(np.int64))

        output = {
            "image":     img,
            "mask":      mask,
            "pathology": s["pathology"],
            "meta": {
                "image_path": s["image_path_str"],
                "grading":    self.grading,
            }
        }
        return output
    


class TripletDataset(Dataset):
    def __init__(self, base_dataset, scan_to_indices):
        self.base_dataset = base_dataset
        self.triplets = []
        for scan_id, indices in scan_to_indices.items():
            sorted_indices = sorted(indices)
            for i in range(1, len(sorted_indices) - 1):
                self.triplets.append((sorted_indices[i - 1], sorted_indices[i], sorted_indices[i + 1]))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        idx_prev, idx_mid, idx_next = self.triplets[idx]
        return {
            "prev": self.base_dataset[idx_prev],
            "mid": self.base_dataset[idx_mid],
            "next": self.base_dataset[idx_next]
        }