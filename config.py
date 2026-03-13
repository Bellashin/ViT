'''
Learning rate를 1e-4로 지정, warmup 진행했더니 학습률이 너무 커져서 최소값을 지나칠 확률이 존재한다고 분석했다.
왜냐하면 epoch4에서 최고 정확도, 최소 손실 값을 보였는데 warm up epoch이 끝나고 나니 성능 향상을 보이지 못하고 있어서 
warmup iteration을 10에서 4로 줄이고, 초기 설정 학습률을 5e-5로 지정하여 안정적으로 최솟값을 찾을 수 있게 설정했다.

warmup iteration은 실험 중 알아낸 hyperparameter이다.

'''
class Config:
    # Data
    img_size     = 224
    patch_size   = 16
    in_channels  = 3
    num_classes  = 30
    
    # Model
    embed_dim    = 256
    num_heads    = 8
    num_encoder   = 6
    mlp_ratio    = 4
    dropout      = 0.1
    
    # Train
    batch_size   = 64
    epochs       = 100
    lr           = 5e-5
    weight_decay = 0.05
    
    # Path
    data_root    = "Dataset"
    save_path    = "checkpoints"