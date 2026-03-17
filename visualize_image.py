import matplotlib.pyplot as plt
from dataset import get_dataset
from config import Config

def show_samples():
    train_loader, _, _, class_idx = get_dataset(
        root=Config.data_root,
        batch_size=16,
        num_workers=0
    )

    # 클래스 이름 (idx → 이름)
    idx_to_class = {v: k for k, v in class_idx.items()}

    # 배치 하나 가져오기
    images, labels = next(iter(train_loader))

    # [-1, 1] → [0, 1] 역정규화
    images = images * 0.5 + 0.5
    images = images.clamp(0, 1)

    # 4x4 그리드로 시각화
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(idx_to_class[labels[i].item()])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('sample.png')  # 파일로 저장
    plt.show()
    print(f"이미지 shape: {images.shape}")
    print(f"픽셀 범위: {images.min():.2f} ~ {images.max():.2f}")

if __name__ == '__main__':
    show_samples()

'''
확인할 수 있는 것들

이미지 shape: torch.Size([16, 3, 224, 224])  → 크기 확인
픽셀 범위: 0.00 ~ 1.00                        → 정규화 확인
클래스 이름이 맞게 붙어있는지               → 레이블 확인
이미지가 찌그러지지 않았는지               → 전처리 확인
'''