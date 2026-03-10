# Quick Start Guide

## 🚀 Getting Started with Image Forgery Detection Pipeline

This guide will help you quickly train and generate predictions for the Kaggle competition.

---

## 📋 Prerequisites

### Required Libraries

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.12.0
- segmentation-models-pytorch
- albumentations
- cv2, pandas, numpy, scikit-learn

### Hardware Requirements

- **Minimum**: GPU with 8GB VRAM
- **Recommended**: Kaggle 2×T4 GPUs (16GB VRAM each)
- **Training Time**: ~3-4 hours for 5 folds

---

## 📁 Data Setup

### Expected Directory Structure

```
/kaggle/input/recod-ai-luc-scientific-image-forgery-detection/
    train_images/
        case_001.tif
        case_002.tif
        ...
    train_masks/
        case_001_1.tif    # Multiple masks per image possible
        case_001_2.tif
        case_002_1.tif
        ...
    supplemental_images/
        ...
    supplemental_masks/
        ...
    test_images/
        test_001.tif
        test_002.tif
        ...
```

---

## 🎯 Quick Start: 3 Simple Steps

### Step 1: Configure

```python
from src.config import Config

config = Config()

# For Kaggle environment
config.update_for_kaggle()

# Adjust hyperparameters if needed
config.EPOCHS = 50
config.BATCH_SIZE = 8
config.LEARNING_RATE = 3e-4
```

### Step 2: Train

```python
from src.train import Trainer
from src.utils import set_seed

# Set seed for reproducibility
set_seed(config.SEED)

# Create trainer
trainer = Trainer(config)

# Train all 5 folds
results = trainer.train_all_folds()

# Results will be saved in: outputs/checkpoints/best_fold{0-4}.pth
```

### Step 3: Generate Submission

```python
from src.inference import generate_submission

# Generate predictions with TTA and ensemble
submission_df = generate_submission(
    config=config,
    fold_indices=None,  # Use all 5 folds
    use_tta=True        # Enable Test Time Augmentation
)

# submission.csv is automatically saved
```

**That's it!** 🎉

---

## 📓 Using the Kaggle Notebook

### Option 1: Run Entire Notebook

1. Upload [notebook.ipynb](notebook.ipynb) to Kaggle
2. Set accelerator to **GPU T4 x2**
3. Click **Run All**
4. Wait ~6 hours for completion
5. Download `submission.csv`

### Option 2: Step-by-Step Execution

The notebook is divided into sections:

1. **Setup** → Install dependencies
2. **Config** → Set hyperparameters
3. **Data Exploration** → Visualize samples
4. **Training** → Train 5-fold CV models
5. **Validation** → Tune thresholds
6. **Inference** → Generate predictions
7. **Visualization** → Check results

Execute cells sequentially for better control.

---

## ⚙️ Configuration Options

All settings in [src/config.py](src/config.py):

### Model Settings

```python
ENCODER_NAME = "efficientnet-b3"  # Backbone
ARCHITECTURE = "Unet"             # Architecture
IMAGE_SIZE = 512                  # Input size
```

**Try different encoders**:
- `efficientnet-b0` (faster, less accurate)
- `efficientnet-b5` (slower, more accurate)
- `resnet50`, `resnet101`

### Training Settings

```python
BATCH_SIZE = 8              # Adjust based on GPU memory
EPOCHS = 50                 # Training epochs
LEARNING_RATE = 3e-4        # Learning rate
N_FOLDS = 5                 # Cross-validation folds
```

### Loss Weights

```python
BCE_WEIGHT = 1.0           # BCE loss weight
DICE_WEIGHT = 1.0          # Dice loss weight
FOCAL_WEIGHT = 0.5         # Focal loss weight
```

### Postprocessing

```python
THRESHOLD = 0.5            # Binarization threshold
MIN_AREA = 100             # Min component size (pixels)
USE_MORPHOLOGY = False     # Morphological operations
```

### TTA Settings

```python
USE_TTA = True
TTA_AUGMENTATIONS = [
    "original",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90"
]
```

---

## 🔧 Common Issues & Solutions

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```python
config.BATCH_SIZE = 4  # Instead of 8
```

**Solution 2**: Use gradient accumulation
```python
config.ACCUMULATION_STEPS = 2
```

**Solution 3**: Reduce image size
```python
config.IMAGE_SIZE = 384  # Instead of 512
```

### Issue: Training Too Slow

**Solution 1**: Use smaller encoder
```python
config.ENCODER_NAME = "efficientnet-b0"
```

**Solution 2**: Reduce workers
```python
config.NUM_WORKERS = 2
```

**Solution 3**: Disable TTA during development
```python
config.USE_TTA = False
```

### Issue: Low Dice Score

**Checklist**:
1. ✅ Using pretrained encoder? (`encoder_weights="imagenet"`)
2. ✅ Proper augmentation? (Check [augmentations.py](src/augmentations.py))
3. ✅ Threshold tuned? (Run validation step)
4. ✅ Class imbalance handled? (`USE_WEIGHTED_SAMPLER=True`)
5. ✅ Enough training epochs? (Check early stopping)

### Issue: RLE Format Error

**Solution**: Verify mask flatten order
```python
from src.rle import test_rle_encoding
test_rle_encoding()  # Run unit tests
```

---

## 📊 Monitoring Training

### During Training

Watch for:
- **Loss decreasing** (train and val)
- **Dice increasing** (validation)
- **No overfitting** (train/val gap < 5%)

### Expected Metrics (per fold)

```
Epoch 10: Train Loss: 0.25, Val Loss: 0.28, Val Dice: 0.68
Epoch 20: Train Loss: 0.18, Val Loss: 0.22, Val Dice: 0.72
Epoch 30: Train Loss: 0.14, Val Loss: 0.20, Val Dice: 0.74
Epoch 40: Train Loss: 0.12, Val Loss: 0.19, Val Dice: 0.75
```

### Red Flags

⚠️ **Val Loss increasing** → Overfitting (reduce epochs)  
⚠️ **Dice < 0.60 after 20 epochs** → Check data/augmentation  
⚠️ **Training crashes** → Reduce batch size  

---

## 🎯 Optimizing for Leaderboard

### Step 1: Baseline (~0.70 Dice)
- Default config
- Single model
- No TTA

### Step 2: Strong Baseline (~0.75 Dice)
- 5-fold CV
- Threshold tuning
- TTA enabled

### Step 3: Advanced (~0.77+ Dice)
- Heavier augmentation
- Better encoder (efficientnet-b5)
- Ensemble tuning
- Advanced postprocessing

---

## 📝 Validation Workflow

### Before Submission

```python
# 1. Train all folds
trainer = Trainer(config)
results = trainer.train_all_folds()

# 2. Validate and tune thresholds
from src.validate import validate_all_folds
val_results = validate_all_folds(config, tune_thr=True)

# 3. Update config with best threshold
best_threshold = np.mean([r['best_threshold'] for r in val_results])
config.THRESHOLD = best_threshold

# 4. Generate submission
submission = generate_submission(config, use_tta=True)

# 5. Sanity check
print(f"Total predictions: {len(submission)}")
print(f"Authentic: {(submission['prediction'] == 'authentic').sum()}")
print(f"Forged: {(submission['prediction'] != 'authentic').sum()}")
```

---

## 🏁 Submission Checklist

Before submitting to Kaggle:

- [ ] All 5 folds trained
- [ ] Thresholds tuned
- [ ] TTA enabled
- [ ] Ensemble using all folds
- [ ] `submission.csv` generated
- [ ] Correct format (case_id, prediction)
- [ ] No missing predictions
- [ ] File size reasonable (< 50MB)

### Verify Submission Format

```python
# Load submission
df = pd.read_csv("submission.csv")

# Check columns
assert list(df.columns) == ['case_id', 'prediction']

# Check for missing values
assert df.isnull().sum().sum() == 0

# Check case_ids match test set
test_ids = sorted([f.stem for f in config.TEST_IMAGES_DIR.glob("*.tif")])
submit_ids = sorted(df['case_id'].tolist())
assert test_ids == submit_ids

print("✅ Submission format validated!")
```

---

## 🐛 Debugging Tips

### Visualize Predictions

```python
from src.utils import visualize_batch, overlay_mask_on_image

# During training
visualize_batch(images, masks, predictions, 
                save_path="debug_train.png")

# On validation set
from src.inference import InferenceEngine
engine = InferenceEngine(config)
engine.load_models([config.CHECKPOINT_DIR / "best_fold0.pth"])

# Predict on sample
pred = engine.predict_single(image, use_tta=False)

# Overlay
result = overlay_mask_on_image(image, pred)
plt.imshow(result)
plt.show()
```

### Check Model Output

```python
model = get_model(...)
model.eval()

with torch.no_grad():
    logits = model(sample_image.unsqueeze(0))
    probs = torch.sigmoid(logits)
    
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"Probs range: [{probs.min():.2f}, {probs.max():.2f}]")
print(f"Mean prob: {probs.mean():.4f}")
```

---

## 🚀 Next Steps

### Improve Performance

1. **Experiment with encoders**
   ```python
   config.ENCODER_NAME = "resnet101"  # Try different backbones
   ```

2. **Increase training time**
   ```python
   config.EPOCHS = 100
   config.EARLY_STOPPING_PATIENCE = 20
   ```

3. **Heavy augmentation**
   ```python
   # In augmentations.py, use "heavy" level
   transform = get_training_augmentation(augmentation_level="heavy")
   ```

4. **Ensemble more models**
   - Train with different random seeds
   - Mix different architectures
   - Average more fold models

### Advanced Techniques

See [ARCHITECTURE.md](ARCHITECTURE.md) Phase 2 section for:
- Transformer backbones
- Attention modules
- Multi-scale training
- External pretraining

---

## 📚 Additional Resources

- **Main Documentation**: [README.md](README.md)
- **Architecture Details**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Source Code**: [src/](src/)
- **Kaggle Notebook**: [notebook.ipynb](notebook.ipynb)

---

## 💡 Pro Tips

1. **Always validate locally** before submitting
2. **Save intermediate results** (predictions, metrics)
3. **Use TTA** for final submission (worth the time)
4. **Monitor GPU usage** to optimize batch size
5. **Ensemble is king** - use all fold models
6. **Reproducibility matters** - fix random seeds

---

## 🎉 Good Luck!

You now have a complete, production-ready pipeline for the competition.

**Expected Timeline**:
- Setup: 10 minutes
- Training: 3-4 hours
- Validation: 30 minutes
- Inference: 1-2 hours
- **Total**: 5-7 hours

**Target Performance**: CV Dice > 0.75, Top 5 potential 🏆

---

**Questions?** Check the detailed documentation in [ARCHITECTURE.md](ARCHITECTURE.md)

**Happy Kaggling! 🚀**
