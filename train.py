import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchinfo import summary

from config import Config
from dataset import get_dataset
from model import ViT
from trainer import Trainer
from utils import set_seed

def main():
    set_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_idx = get_dataset(
        root=Config.data_root,
        batch_size=Config.batch_size,
        num_workers = 4
    )

    model= ViT(
        in_channels=Config.in_channels,
        embedded_size=Config.embed_dim,
        num_encoders=Config.num_enocder,
        patch_size=Config.patch_size,
        num_classes=Config.num_classes,
        num_heads=Config.num_heads,
        img_size=Config.img_size
    ).to(device)

    summary(model, input_size=(1,3,Config.img_size, Config.img_size), device = device)

    optimizer=Adam(
        model.parameters(),
        lr=Config.lr,
        betas=(0.9, 0.999),
        weight_decay=Config.weight_decay
    )

    warmup_scheduler=LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=10
    )
    decay_scheduler=LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=Config.epochs-10
    )
    scheduler=SequentialLR(
        optimizer,
        schedulers =[warmup_scheduler, decay_scheduler],
        milestones=[10]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=Config,
    )

    trainer.best()

if __name__ == '__main__':
    main()


