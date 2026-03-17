import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from config import Config
from dataset import get_dataset
from model import ViT
from trainer import Trainer
from utils import set_seed

CHECKPOINT_PATH ='checkpoints/best_model.pth'
EXTRA_EPOCHS    = 50   # 추가로 학습할 epoch 수
START_LR        = 1e-7      # 마지막 학습률에서 더 낮춘 학습률로 
END_LR_FACTOR   = 1     # 학습률은 고정해서

def main():
    set_seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_idx = get_dataset(
        root=Config.data_root,
        batch_size=Config.batch_size,
        num_workers=2
    )

    model = ViT(
        in_channels=Config.in_channels,
        embedded_size=Config.embed_dim,
        num_encoders=Config.num_encoder,
        patch_size=Config.patch_size,
        num_classes=Config.num_classes,
        num_heads=Config.num_heads,
        img_size=Config.img_size,
        dropout=Config.dropout
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=START_LR,
        weight_decay=Config.weight_decay
    )

    # 체크포인트 로드
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['val_acc']
    print(f"체크포인트 로드 완료 | Epoch {start_epoch+1} | Val Acc: {best_val_acc:.4f}")

    # 이어서 학습할 때는 warmup 없이 decay만
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=EXTRA_EPOCHS
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Config를 동적으로 수정해서 Trainer에 전달
    Config.epochs = EXTRA_EPOCHS

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=Config,
        start_epoch=start_epoch + 1,       # 출력용 epoch 번호
    )

    trainer.best()

if __name__ == '__main__':
    main()