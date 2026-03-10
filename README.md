# BioForgeNet
> **Scientific Image Forgery Detection — Kaggle Competition Solution**

A modular, production-ready PyTorch pipeline for detecting copy-move forgeries in biomedical images.

## 🎯 Competition

**Recod.ai/LUC – Scientific Image Forgery Detection**

Detect and segment copy-move forgeries in biomedical images at pixel level.

## 🏗️ Architecture

### Phase 1: Strong U-Net Baseline
- **Encoder**: EfficientNet-B3 (ImageNet pretrained)
- **Architecture**: U-Net with skip connections
- **Loss**: Hybrid (BCE + Dice + Focal)
- **Augmentation**: Albumentations pipeline
- **Training**: 5-Fold Cross-Validation

### Phase 2: Advanced Optimizations
- Test Time Augmentation (TTA)
- Multi-model ensemble
- Postprocessing & threshold tuning
- Class imbalance handling

## 📂 Project Structure

```
/src
    config.py              # Central configuration
    dataset.py             # Data loading & preprocessing
    augmentations.py       # Albumentations pipeline
    models.py              # Model architectures (SMP)
    losses.py              # Hybrid loss functions
    metrics.py             # Evaluation metrics
    train.py               # Training loop with CV
    validate.py            # Validation & threshold tuning
    inference.py           # Inference with TTA & ensemble
    postprocess.py         # Postprocessing utilities
    rle.py                 # RLE encoding for submission
    utils.py               # Helper functions
notebook.ipynb             # Kaggle entry point
```

## 🚀 Quick Start

### Training

```python
from src.config import Config
from src.train import Trainer

# Configure
config = Config()
config.EPOCHS = 50
config.BATCH_SIZE = 8

# Train all folds
trainer = Trainer(config)
results = trainer.train_all_folds()
```

### Inference

```python
from src.config import Config
from src.inference import generate_submission

# Generate submission
config = Config()
submission_df = generate_submission(
    config=config,
    fold_indices=None,  # Use all folds
    use_tta=True
)
```

## 📊 Key Features

### Data Pipeline
- ✅ Multiple masks per image → merged into single binary mask
- ✅ Handles authentic images (no forgery)
- ✅ Stratified K-fold cross-validation
- ✅ Weighted sampling for class imbalance

### Model
- ✅ Segmentation Models PyTorch (SMP)
- ✅ Multiple architecture options (U-Net, U-Net++, FPN, DeepLabV3+)
- ✅ Pretrained encoders (EfficientNet, ResNet, etc.)
- ✅ Optional attention modules

### Loss Function
- ✅ Hybrid: BCE + Dice + Focal
- ✅ Handles empty masks safely
- ✅ Configurable weights

### Training
- ✅ Mixed precision (AMP)
- ✅ Gradient clipping
- ✅ CosineAnnealingLR scheduler
- ✅ Early stopping
- ✅ Model checkpointing

### Inference
- ✅ Test Time Augmentation (TTA)
- ✅ Multi-model ensemble
- ✅ Configurable postprocessing
- ✅ Automatic RLE encoding

### Postprocessing
- ✅ Threshold tuning
- ✅ Small component removal
- ✅ Morphological operations
- ✅ Hole filling

## 📈 Expected Performance

- **Baseline**: CV Dice > 0.70
- **Optimized**: CV Dice > 0.75
- **With TTA**: CV Dice > 0.77

## 🔧 Configuration

All hyperparameters in `src/config.py`:

```python
IMAGE_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4
ENCODER_NAME = "efficientnet-b3"
N_FOLDS = 5
USE_TTA = True
```

## 📝 Kaggle Notebook

The main notebook (`notebook.ipynb`) includes:

1. Environment setup
2. Data exploration
3. Training pipeline
4. Validation & threshold tuning
5. Inference with TTA
6. Submission generation

## 🎓 Advanced Features (Phase 2)

Ready for enhancement:
- Transformer backbones (Swin)
- Self-correlation modules
- Multi-scale training
- External dataset pretraining
- Advanced ensembling

## 📦 Dependencies

- torch >= 1.12.0
- torchvision >= 0.13.0
- segmentation-models-pytorch
- albumentations
- opencv-python
- pandas
- numpy
- scikit-learn
- tqdm
- matplotlib

## 🏆 Reproducibility

- Fixed random seeds
- Deterministic algorithms
- Documented hyperparameters
- Version-controlled code

## 📧 Notes

- Optimized for Kaggle 2x T4 GPUs (16GB VRAM)
- Training time: ~3-4 hours for 5 folds
- Inference time: ~30-45 minutes with TTA

---

**Built for Top 5 Kaggle Competition Performance** 🥇
