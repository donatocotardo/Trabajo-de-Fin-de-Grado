import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

def compute_sdf(binary_mask):
    # Distancia firmada de una máscara binaria: dentro positivo, fuera negativo.
    if not np.any(binary_mask):
        return -np.ones_like(binary_mask, dtype=float) * 1e5
    if np.all(binary_mask):
        return np.ones_like(binary_mask, dtype=float) * 1e5

    # Se mide cuánto se aleja cada píxel del borde por dentro y por fuera
    dist_inside = distance_transform_edt(binary_mask)
    dist_outside = distance_transform_edt(~binary_mask)
    return dist_inside - dist_outside

def interpolate_masks(mask_prev, mask_next, num_classes=6):
    device = mask_prev.device
    
    needs_squeeze = False
    # Si entra una sola máscara 2D, se trata como lote de tamaño 1 y luego se devuelve igual.
    if mask_prev.ndim == 2:
        mask_prev = mask_prev.unsqueeze(0)
        mask_next = mask_next.unsqueeze(0)
        needs_squeeze = True
        
    b, h, w = mask_prev.shape
    # se pasa a numpy para hacer la interpolación clase a clase
    np_prev = mask_prev.cpu().numpy()
    np_next = mask_next.cpu().numpy()
    np_interp = np.zeros_like(np_prev)

    for i in range(b):
        # Aquí se acumula la máscara final de cada pixel
        sum_binary_masks = np.zeros((h, w), dtype=np.int32)
        
        # Interpolación pura sin tocar el fondo manualmente
        for k in range(num_classes - 1):
            # Cada clase se reconstruye como una máscara binaria independiente
            bin_prev = (np_prev[i] > k)
            bin_next = (np_next[i] > k)

            # se interpola el contorno entre ambas máscaras y me quedo con el punto medio
            sdf_prev = compute_sdf(bin_prev)
            sdf_next = compute_sdf(bin_next)

            sdf_mid = (sdf_prev + sdf_next) / 2.0
            bin_mid = (sdf_mid > 0)
            
            # se suman las clases que quedan activas para reconstruir la máscara entera
            sum_binary_masks += bin_mid.astype(np.int32)
                    
        np_interp[i] = sum_binary_masks

    interp_tensor = torch.tensor(np_interp, device=device, dtype=torch.long)
    
    if needs_squeeze:
         interp_tensor = interp_tensor.squeeze(0)
         
    return interp_tensor