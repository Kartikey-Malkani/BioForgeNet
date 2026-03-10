"""
Training loop with K-fold cross-validation, mixed precision, and early stopping
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config
from models import get_model
from losses import get_loss_function
from metrics import MetricsTracker, compute_all_metrics
from dataset import ForgeryDataset, prepare_folds, create_weighted_sampler, get_dataloader
from augmentations import get_training_augmentation, get_validation_augmentation
from utils import set_seed, get_device, save_checkpoint, load_checkpoint, AverageMeter, format_time


class Trainer:
    """
    Training class with cross-validation support
    """
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # Create directories
        config.create_directories()
        
        # Set seed
        set_seed(config.SEED, config.DETERMINISTIC)
        
        # Prepare folds
        print("\nPreparing cross-validation folds...")
        self.folds = prepare_folds(
            image_dir=config.TRAIN_IMAGES_DIR,
            mask_dir=config.TRAIN_MASKS_DIR,
            n_folds=config.N_FOLDS,
            seed=config.SEED,
            stratify=config.STRATIFY
        )
    
    def train_fold(self, fold: int, resume: bool = True):
        """
        Train a single fold
        
        Args:
            fold: Fold index
        
        Returns:
            Best validation metrics
        """
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold}")
        print(f"{'='*60}")
        
        # Get train and validation case IDs
        train_ids, val_ids = self.folds[fold]
        
        # Create datasets
        train_transform = get_training_augmentation(
            image_size=self.config.IMAGE_SIZE,
            augmentation_level="medium"
        )
        val_transform = get_validation_augmentation(
            image_size=self.config.IMAGE_SIZE
        )
        
        train_dataset = ForgeryDataset(
            image_dir=self.config.TRAIN_IMAGES_DIR,
            mask_dir=self.config.TRAIN_MASKS_DIR,
            case_ids=train_ids,
            transform=train_transform,
            mode="train"
        )
        
        val_dataset = ForgeryDataset(
            image_dir=self.config.TRAIN_IMAGES_DIR,
            mask_dir=self.config.TRAIN_MASKS_DIR,
            case_ids=val_ids,
            transform=val_transform,
            mode="val"
        )
        
        # Create dataloaders
        train_sampler = None
        if self.config.USE_WEIGHTED_SAMPLER:
            train_sampler = create_weighted_sampler(train_dataset, weight_ratio=2.0)
        
        train_loader = get_dataloader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=(train_sampler is None),
            num_workers=self.config.NUM_WORKERS,
            sampler=train_sampler
        )
        
        val_loader = get_dataloader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        # Create model
        model = get_model(
            architecture=self.config.ARCHITECTURE,
            encoder_name=self.config.ENCODER_NAME,
            encoder_weights=self.config.ENCODER_WEIGHTS,
            in_channels=self.config.IN_CHANNELS,
            classes=self.config.NUM_CLASSES,
            use_attention=self.config.USE_ATTENTION
        )
        model = model.to(self.device)
        
        # Loss function
        criterion = get_loss_function(self.config).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Scheduler
        if self.config.SCHEDULER == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.T_MAX,
                eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Mixed precision scaler
        scaler = GradScaler() if self.config.USE_AMP else None
        
        # Training state
        checkpoint_path = self.config.CHECKPOINT_DIR / f"best_fold{fold}.pth"
        last_checkpoint_path = self.config.CHECKPOINT_DIR / f"last_fold{fold}.pth"
        start_epoch = 1
        best_dice = 0.0
        patience_counter = 0
        
        # Resume from checkpoint if available
        if resume:
            resume_path = None
            if last_checkpoint_path.exists():
                resume_path = last_checkpoint_path
            elif checkpoint_path.exists():
                resume_path = checkpoint_path
            
            if resume_path is not None:
                checkpoint_info = load_checkpoint(resume_path, model, optimizer, scheduler)
                start_epoch = checkpoint_info.get('epoch', 0) + 1
                metrics = checkpoint_info.get('metrics', {})
                best_dice = metrics.get('val_dice', metrics.get('dice', 0.0))
                print(f"Resuming fold {fold} from epoch {start_epoch} (best_dice={best_dice:.4f})")
        
        if start_epoch > self.config.EPOCHS:
            print(f"Fold {fold} already completed (start_epoch={start_epoch} > EPOCHS={self.config.EPOCHS})")
            return {
                'fold': fold,
                'best_dice': best_dice,
                'best_metrics': {'dice': best_dice}
            }
        
        # Training loop
        for epoch in range(start_epoch, self.config.EPOCHS + 1):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss, train_metrics = self.train_epoch(
                model, train_loader, criterion, optimizer, scaler, epoch
            )
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(
                model, val_loader, criterion, epoch
            )
            
            # Update scheduler
            if self.config.SCHEDULER == "CosineAnnealingLR":
                scheduler.step()
            else:
                scheduler.step(val_metrics['dice'])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{self.config.EPOCHS} - Time: {format_time(epoch_time)}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            
            # Save last checkpoint each epoch (for resume)
            save_checkpoint(
                model, optimizer, scheduler, epoch, fold,
                {'val_dice': val_metrics['dice'], 'val_metrics': val_metrics},
                last_checkpoint_path
            )
            
            # Save best model
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                patience_counter = 0
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch, fold,
                    {'val_dice': best_dice, 'val_metrics': val_metrics},
                    checkpoint_path
                )
                print(f"✓ Best model saved! Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print(f"\nFold {fold} completed. Best Dice: {best_dice:.4f}")
        
        return {
            'fold': fold,
            'best_dice': best_dice,
            'best_metrics': val_metrics
        }
    
    def train_epoch(self, model, loader, criterion, optimizer, scaler, epoch):
        """Train for one epoch"""
        model.train()
        
        loss_meter = AverageMeter()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss, loss_dict = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.GRADIENT_CLIP_VAL > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.GRADIENT_CLIP_VAL
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)
                
                optimizer.zero_grad()
                loss.backward()
                
                if self.config.GRADIENT_CLIP_VAL > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.GRADIENT_CLIP_VAL
                    )
                
                optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            with torch.no_grad():
                metrics_tracker.update(outputs.detach(), masks, threshold=self.config.THRESHOLD)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss_meter.avg})
        
        avg_metrics = metrics_tracker.get_average()
        return loss_meter.avg, avg_metrics
    
    def validate_epoch(self, model, loader, criterion, epoch):
        """Validate for one epoch"""
        model.eval()
        
        loss_meter = AverageMeter()
        metrics_tracker = MetricsTracker()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss, loss_dict = criterion(outputs, masks)
                
                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                metrics_tracker.update(outputs, masks, threshold=self.config.THRESHOLD)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss_meter.avg})
        
        avg_metrics = metrics_tracker.get_average()
        return loss_meter.avg, avg_metrics
    
    def train_all_folds(self):
        """Train all folds and report results"""
        print("\n" + "="*60)
        print("STARTING CROSS-VALIDATION TRAINING")
        print("="*60)
        
        fold_results = []
        
        for fold in self.config.TRAIN_FOLDS:
            result = self.train_fold(fold)
            fold_results.append(result)
        
        # Print summary
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        
        mean_dice = sum([r['best_dice'] for r in fold_results]) / len(fold_results)
        
        for result in fold_results:
            print(f"Fold {result['fold']}: Dice = {result['best_dice']:.4f}")
        
        print(f"\nMean CV Dice: {mean_dice:.4f}")
        print("="*60)
        
        return fold_results


def main():
    """Main training function"""
    # Load config
    config = Config()
    
    # Print configuration
    config.print_config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train all folds
    results = trainer.train_all_folds()
    
    print("\nTraining completed!")
    return results


if __name__ == "__main__":
    main()
