"""
Model architectures for image forgery detection
Uses segmentation_models_pytorch (SMP) with optional attention modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class AttentionBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEBlock(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation block
    Combines channel attention and spatial attention
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        channel_att = self.cSE(x) * x
        spatial_att = self.sSE(x) * x
        return channel_att + spatial_att


def get_model(
    architecture: str = "Unet",
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: str = None,
    use_attention: bool = False
):
    """
    Create segmentation model
    
    Args:
        architecture: Architecture name (Unet, UnetPlusPlus, FPN, DeepLabV3Plus, etc.)
        encoder_name: Encoder backbone (efficientnet-b3, resnet50, etc.)
        encoder_weights: Pretrained weights (imagenet, None)
        in_channels: Number of input channels
        classes: Number of output classes
        activation: Output activation (None for logits)
        use_attention: Whether to add attention modules
    
    Returns:
        PyTorch model
    """
    
    # Get architecture class from SMP
    if architecture == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif architecture == "MAnet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Optionally add attention (this would require modifying the decoder)
    # For now, we use SMP's built-in attention in some architectures like MAnet
    if use_attention and architecture not in ["MAnet"]:
        print("Warning: Custom attention not implemented. Use MAnet for built-in attention.")
    
    return model


class ForgeryDetectionModel(nn.Module):
    """
    Wrapper model with optional enhancements
    """
    
    def __init__(
        self,
        architecture: str = "Unet",
        encoder_name: str = "efficientnet-b3",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.model = get_model(
            architecture=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # We'll apply sigmoid in loss/inference
            use_attention=use_attention
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Logits [B, 1, H, W]
        """
        return self.model(x)


class EnsembleModel(nn.Module):
    """
    Ensemble multiple models for inference
    """
    
    def __init__(self, models: list, method: str = "mean"):
        """
        Args:
            models: List of PyTorch models
            method: Ensemble method ("mean" or "max")
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
    
    def forward(self, x):
        """
        Forward pass through all models
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Averaged/max logits [B, 1, H, W]
        """
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        # Stack and aggregate
        outputs = torch.stack(outputs)
        
        if self.method == "mean":
            return outputs.mean(dim=0)
        elif self.method == "max":
            return outputs.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")


def load_model_checkpoint(checkpoint_path, device="cuda"):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model configuration if available
    # For now, we'll use default config
    # In production, you'd save model config in checkpoint
    
    from config import Config
    
    model = get_model(
        architecture=Config.ARCHITECTURE,
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # Don't load ImageNet weights again
        in_channels=Config.IN_CHANNELS,
        classes=Config.NUM_CLASSES,
        activation=None
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model


def test_model():
    """Test model creation and forward pass"""
    print("Testing model creation...")
    
    # Test basic model creation
    model = get_model(
        architecture="Unet",
        encoder_name="efficientnet-b0",
        encoder_weights=None,  # Don't download for testing
        in_channels=3,
        classes=1
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 1, 512, 512), "Output shape mismatch"
    print("Model test passed! ✓")


if __name__ == "__main__":
    test_model()
