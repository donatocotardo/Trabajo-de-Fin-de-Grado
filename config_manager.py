import os
import json
import torch

def load_config_interactively():
    config_path = "args.json"
    if not os.path.exists(config_path):
        print(f"Error: No se encuentra {config_path}")
        return None
    with open(config_path, 'r') as f:
        configs = json.load(f)
    print("\nExperimentos disponibles en el JSON")
    for i, c in enumerate(configs):
        print(f"[{i}] {c.get('name', 'Exp')} - Loss: {c['loss_type']}, LR: {c['lr']}")
    while True:
        try:
            res = input("\nSelecciona el índice del experimento a ejecutar: ")
            idx = int(res)
            return configs[idx]
        except (ValueError, IndexError):
            print("Índice no válido. Inténtalo de nuevo.")

def load_config_from_cli(args):
    return {
        'name':        args.name,
        'loss_type':   args.loss_type,
        'lr':          float(args.lr),
        'batch_size':  int(args.batch_size),
        'epochs':      int(args.epochs),
        'resume_ckpt': args.resume_ckpt
    }

def load_checkpoint(ckpt_path, model, optimizer, device, config_data):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint inválido: falta la clave 'state_dict'.")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        json_lr = float(config_data["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = json_lr
    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    train_losses = list(ckpt.get("train_losses", []))
    print(f"\nCheckpoint cargado: {ckpt_path}")
    return start_epoch, global_step, train_losses