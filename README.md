# ViT
Vision Transformer pre-trained with Plants Classification Datasets <br/>
Datasets url : https://www.kaggle.com/datasets/marquis03/plants-classification


Vision Transformer needs LARGE datasets to pre-train, but I use Plants Classification Datasets which contain 30 types of plants images, including 21000 training images, 3000 validation images and 6000 test images, with a total data size of 1.48GB.

## Project Structure

```bash
.
├── checkpoints/                 # Saved model checkpoints
├── Dataset/
│   ├── train/                   # Training images
│   ├── val/                     # Validation images
│   ├── test/                    # Test images
│   ├── train.csv                # Training labels
│   ├── val.csv                  # Validation labels
│   └── test.csv                 # Test labels
│
├── Visualize/                   # Visualization outputs
│
├── config.py                    # Training configuration
├── dataset.py                   # Dataset loader
├── model.py                     # Vision Transformer model
├── trainer.py                   # Training logic
├── train.py                     # Training entry script
├── extra_train.py               # Additional training experiments (50 epochs)
│
├── utils.py                     # Utility functions
├── visualize_image.py           # Preprocessed image visualization
├── visualize_train_graph.py     # Training curve visualization
│
└── README.md                    # Project documentation
```
