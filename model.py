import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedded_size=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.Norm1 = nn.LayerNorm(embedded_size)
        self.attention = nn.MultiheadAttention(embedded_size, num_heads, dropout, batch_first=True)
        self.Norm2 = nn.LayerNorm(embedded_size)
        self.enc_mlp = nn.Sequential(
            nn.Linear(embedded_size, embedded_size*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedded_size*4, embedded_size),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.Norm1(x)
        x = x + self.attention(x, x, x)[0]
        x = self.Norm2(x)
        x = x + self.enc_mlp(x)
        
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=3, embedded_size=768, num_encoders=4, patch_size=16, num_classes=30, num_heads=8, img_size=224):
        super().__init__()
        self.img_size=img_size
        num_tokens = (img_size*img_size)//(patch_size**2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedded_size), requires_grad=True)
        self.patch_size=patch_size
        self.embedded_size=embedded_size
        # patch embedding : 데이터를 의미있는 임베딩으로 바꾼다 
        self.patch_embedding=nn.Conv2d(in_channels, embedded_size, kernel_size=patch_size, stride=patch_size)
        # position embedding : 처음엔 랜덤한 값으로 시작하지만 학습이 끝나면 위치 특성이 담긴 지도 역할을 하게 된다 (cls 토큰 포함)
        self.postion_embedding = nn.Parameter(torch.randn((num_tokens+1, embedded_size)), requires_grad=True)
        self.in_channels=in_channels
        self.num_encoders=num_encoders
        self.num_classes=num_classes
        self.num_heads=num_heads

        self.encdoers = nn.ModuleList([
            Encoder(embedded_size=embedded_size, num_heads=num_heads) for _ in range(num_encoders)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedded_size),
            nn.Linear(embedded_size, embedded_size),
            nn.Tanh(),
            nn.Linear(embedded_size, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.postion_embedding
        for encoder in self.encdoers:
            x = encoder(x)
        x = x[:,0,:]
        x = self.mlp_head(x)
        return x





