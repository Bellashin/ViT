
class Config:
    # Data
    img_size     = 224
    patch_size   = 16
    in_channels  = 3
    num_classes  = 30
    
    # Model
    embed_dim    = 768
    num_heads    = 8
    num_enocder   = 4
    mlp_ratio    = 4
    dropout      = 0.1
    
    # Train
    batch_size   = 64
    epochs       = 100
    lr           = 1e-3
    weight_decay = 0.05
    
    # Path
    data_root    = "Dataset"
    save_path    = "checkpoints"