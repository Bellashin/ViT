"""
사용법:
1. LOG 변수 안에 터미널 출력 로그를 붙여넣기
2. python plot_log.py 실행
3. training_history.png 로 저장됨
"""

import re
import matplotlib.pyplot as plt

LOG = """
Epoch [99/50] LR: 0.00000010 | Train Loss: 1.1992 | Train Acc: 0.7917 | Val Loss: 1.4168 | Val Acc: 0.7309
Epoch [100/50] LR: 0.00000010 | Train Loss: 1.2054 | Train Acc: 0.7898 | Val Loss: 1.4163 | Val Acc: 0.7312
Epoch [101/50] LR: 0.00000010 | Train Loss: 1.2008 | Train Acc: 0.7944 | Val Loss: 1.4166 | Val Acc: 0.7299
Epoch [102/50] LR: 0.00000010 | Train Loss: 1.2041 | Train Acc: 0.7899 | Val Loss: 1.4169 | Val Acc: 0.7309
Epoch [103/50] LR: 0.00000010 | Train Loss: 1.2031 | Train Acc: 0.7908 | Val Loss: 1.4171 | Val Acc: 0.7286
Epoch [104/50] LR: 0.00000010 | Train Loss: 1.2024 | Train Acc: 0.7940 | Val Loss: 1.4172 | Val Acc: 0.7286
Epoch [105/50] LR: 0.00000010 | Train Loss: 1.2106 | Train Acc: 0.7889 | Val Loss: 1.4165 | Val Acc: 0.7286
Epoch [106/50] LR: 0.00000010 | Train Loss: 1.2048 | Train Acc: 0.7941 | Val Loss: 1.4152 | Val Acc: 0.7309
Epoch [107/50] LR: 0.00000010 | Train Loss: 1.1989 | Train Acc: 0.7912 | Val Loss: 1.4159 | Val Acc: 0.7286
Epoch [108/50] LR: 0.00000010 | Train Loss: 1.2066 | Train Acc: 0.7880 | Val Loss: 1.4170 | Val Acc: 0.7296
Epoch [109/50] LR: 0.00000010 | Train Loss: 1.2023 | Train Acc: 0.7927 | Val Loss: 1.4163 | Val Acc: 0.7282
Epoch [110/50] LR: 0.00000010 | Train Loss: 1.2035 | Train Acc: 0.7905 | Val Loss: 1.4154 | Val Acc: 0.7289
Epoch [111/50] LR: 0.00000010 | Train Loss: 1.1990 | Train Acc: 0.7931 | Val Loss: 1.4170 | Val Acc: 0.7289
Epoch [112/50] LR: 0.00000010 | Train Loss: 1.2042 | Train Acc: 0.7917 | Val Loss: 1.4169 | Val Acc: 0.7292
Epoch [113/50] LR: 0.00000010 | Train Loss: 1.2018 | Train Acc: 0.7928 | Val Loss: 1.4171 | Val Acc: 0.7292
Epoch [114/50] LR: 0.00000010 | Train Loss: 1.2018 | Train Acc: 0.7960 | Val Loss: 1.4162 | Val Acc: 0.7299
Epoch [115/50] LR: 0.00000010 | Train Loss: 1.2049 | Train Acc: 0.7915 | Val Loss: 1.4172 | Val Acc: 0.7292
Epoch [116/50] LR: 0.00000010 | Train Loss: 1.1973 | Train Acc: 0.7940 | Val Loss: 1.4162 | Val Acc: 0.7286
Epoch [117/50] LR: 0.00000010 | Train Loss: 1.2037 | Train Acc: 0.7931 | Val Loss: 1.4170 | Val Acc: 0.7292
Epoch [118/50] LR: 0.00000010 | Train Loss: 1.1980 | Train Acc: 0.7916 | Val Loss: 1.4162 | Val Acc: 0.7292
Epoch [119/50] LR: 0.00000010 | Train Loss: 1.2101 | Train Acc: 0.7878 | Val Loss: 1.4176 | Val Acc: 0.7282
Epoch [120/50] LR: 0.00000010 | Train Loss: 1.2021 | Train Acc: 0.7913 | Val Loss: 1.4158 | Val Acc: 0.7296
Epoch [121/50] LR: 0.00000010 | Train Loss: 1.2037 | Train Acc: 0.7911 | Val Loss: 1.4153 | Val Acc: 0.7299
Epoch [122/50] LR: 0.00000010 | Train Loss: 1.2064 | Train Acc: 0.7892 | Val Loss: 1.4152 | Val Acc: 0.7292
Epoch [123/50] LR: 0.00000010 | Train Loss: 1.2013 | Train Acc: 0.7911 | Val Loss: 1.4170 | Val Acc: 0.7296
Epoch [124/50] LR: 0.00000010 | Train Loss: 1.2038 | Train Acc: 0.7931 | Val Loss: 1.4170 | Val Acc: 0.7296
Epoch [125/50] LR: 0.00000010 | Train Loss: 1.2051 | Train Acc: 0.7913 | Val Loss: 1.4159 | Val Acc: 0.7292
Epoch [126/50] LR: 0.00000010 | Train Loss: 1.2004 | Train Acc: 0.7898 | Val Loss: 1.4166 | Val Acc: 0.7296
Epoch [127/50] LR: 0.00000010 | Train Loss: 1.1984 | Train Acc: 0.7962 | Val Loss: 1.4172 | Val Acc: 0.7299
Epoch [128/50] LR: 0.00000010 | Train Loss: 1.2054 | Train Acc: 0.7896 | Val Loss: 1.4176 | Val Acc: 0.7286
Epoch [129/50] LR: 0.00000010 | Train Loss: 1.2085 | Train Acc: 0.7916 | Val Loss: 1.4175 | Val Acc: 0.7296
Epoch [130/50] LR: 0.00000010 | Train Loss: 1.2008 | Train Acc: 0.7942 | Val Loss: 1.4181 | Val Acc: 0.7289
Epoch [131/50] LR: 0.00000010 | Train Loss: 1.2072 | Train Acc: 0.7872 | Val Loss: 1.4162 | Val Acc: 0.7289
Epoch [132/50] LR: 0.00000010 | Train Loss: 1.1929 | Train Acc: 0.7927 | Val Loss: 1.4168 | Val Acc: 0.7289
Epoch [133/50] LR: 0.00000010 | Train Loss: 1.1968 | Train Acc: 0.7926 | Val Loss: 1.4159 | Val Acc: 0.7282
Epoch [134/50] LR: 0.00000010 | Train Loss: 1.1988 | Train Acc: 0.7929 | Val Loss: 1.4161 | Val Acc: 0.7292
Epoch [135/50] LR: 0.00000010 | Train Loss: 1.1975 | Train Acc: 0.7949 | Val Loss: 1.4165 | Val Acc: 0.7292
Epoch [136/50] LR: 0.00000010 | Train Loss: 1.2075 | Train Acc: 0.7886 | Val Loss: 1.4170 | Val Acc: 0.7279
Epoch [137/50] LR: 0.00000010 | Train Loss: 1.1943 | Train Acc: 0.7942 | Val Loss: 1.4162 | Val Acc: 0.7286
Epoch [138/50] LR: 0.00000010 | Train Loss: 1.2009 | Train Acc: 0.7907 | Val Loss: 1.4169 | Val Acc: 0.7289
Epoch [139/50] LR: 0.00000010 | Train Loss: 1.2035 | Train Acc: 0.7890 | Val Loss: 1.4163 | Val Acc: 0.7296
Epoch [140/50] LR: 0.00000010 | Train Loss: 1.1958 | Train Acc: 0.7964 | Val Loss: 1.4163 | Val Acc: 0.7289
Epoch [141/50] LR: 0.00000010 | Train Loss: 1.2002 | Train Acc: 0.7936 | Val Loss: 1.4152 | Val Acc: 0.7299
Epoch [142/50] LR: 0.00000010 | Train Loss: 1.2008 | Train Acc: 0.7914 | Val Loss: 1.4166 | Val Acc: 0.7289
Epoch [143/50] LR: 0.00000010 | Train Loss: 1.1963 | Train Acc: 0.7929 | Val Loss: 1.4167 | Val Acc: 0.7282
Epoch [144/50] LR: 0.00000010 | Train Loss: 1.1904 | Train Acc: 0.7987 | Val Loss: 1.4157 | Val Acc: 0.7279
Epoch [145/50] LR: 0.00000010 | Train Loss: 1.2085 | Train Acc: 0.7916 | Val Loss: 1.4141 | Val Acc: 0.7282
Epoch [146/50] LR: 0.00000010 | Train Loss: 1.2096 | Train Acc: 0.7869 | Val Loss: 1.4142 | Val Acc: 0.7296
Epoch [147/50] LR: 0.00000010 | Train Loss: 1.2043 | Train Acc: 0.7928 | Val Loss: 1.4151 | Val Acc: 0.7296
Epoch [148/50] LR: 0.00000010 | Train Loss: 1.1964 | Train Acc: 0.7954 | Val Loss: 1.4150 | Val Acc: 0.7289
"""

def parse_log(log: str):
    pattern = (
        r"Epoch \[(\d+)/\d+\].*?"
        r"Train Loss: ([\d.]+).*?"
        r"Train Acc: ([\d.]+).*?"
        r"Val Loss: ([\d.]+).*?"
        r"Val Acc: ([\d.]+)"
    )
    epochs, train_losses, train_accs, val_losses, val_accs = [], [], [], [], []

    for m in re.finditer(pattern, log):
        epochs.append(int(m.group(1)))
        train_losses.append(float(m.group(2)))
        train_accs.append(float(m.group(3)))
        val_losses.append(float(m.group(4)))
        val_accs.append(float(m.group(5)))

    return epochs, train_losses, train_accs, val_losses, val_accs


def plot(epochs, train_losses, train_accs, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses,   label='Val Loss',   linewidth=2)
    ax1.set_title('Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs,   label='Val Acc',   linewidth=2)
    ax2.set_title('Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('extra_training_history.png', dpi=150)
    plt.close()
    print(f"저장 완료: extra_training_history.png")
    print(f"  최고 Val Acc: {max(val_accs):.4f} (Epoch {epochs[val_accs.index(max(val_accs))]})")
    print(f"  최저 Val Loss: {min(val_losses):.4f} (Epoch {epochs[val_losses.index(min(val_losses))]})")


if __name__ == '__main__':
    epochs, train_losses, train_accs, val_losses, val_accs = parse_log(LOG)
    print(f"파싱된 epoch 수: {len(epochs)}")
    plot(epochs, train_losses, train_accs, val_losses, val_accs)