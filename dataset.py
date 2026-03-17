'''
ViT/Dataset/train/aloevera/aloevera0.jpg
이런 형식의 파일 구조는 PyTorch ImageFolder와 완전히 호환이 된다
폴더 이름이 자동으로 레이블로 변환된다
'''

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import Config

def get_dataset(root, batch_size, num_workers):
    train_transform = transforms.Compose([
        # 크롭하는 스케일을 높여 물체가 아닌 배경만 잡히는 문제 해결하려고 함 (trade off 존재: 배경 문제 - 증강 효과)
        transforms.1.o(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        ## Extra augmentation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ImageFolder(root=os.path.join(Config.data_root,'train'), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(Config.data_root,'val'), transform=val_transform)
    test_dataset = ImageFolder(root=os.path.join(Config.data_root, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    class_idx = train_dataset.class_to_idx

    return train_loader, val_loader, test_loader, class_idx
