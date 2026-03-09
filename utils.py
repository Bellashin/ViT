import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, label):
    pred=output.argmax(dim=1)
    return (pred==label).float().mean().item()

def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        'epoch':epoch,
        'model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'val_acc':val_acc
    }, f"{save_path}/best_model.pth")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['epoch'], checkpoint['val_acc']

