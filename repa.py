"""
REPA (REPresentation Alignment) utilities for DiT training.
Based on: https://github.com/sihyun-yu/REPA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


# DINOv2 default normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_mlp(hidden_size, projector_dim, z_dim):
    """Build a projector MLP to map DiT features to encoder feature space."""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


@torch.no_grad()
def load_encoder(encoder_type="dinov2", model_size="s", device="cuda"):
    """
    Load a pre-trained encoder for feature extraction.
    
    Args:
        encoder_type: Type of encoder ("dinov2", "dinov2_reg")
        model_size: Model size ("s", "b", "l", "g")
        device: Device to load model on
    
    Returns:
        encoder: The encoder model
        embed_dim: The embedding dimension of the encoder
    """
    if "dinov2" in encoder_type:
        if "reg" in encoder_type:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14_reg')
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
        
        embed_dim = encoder.embed_dim
        encoder = encoder.to(device)
        encoder.eval()
        
        return encoder, embed_dim
    else:
        raise NotImplementedError(f"Encoder type {encoder_type} not supported")


def preprocess_for_encoder(x, encoder_type="dinov2", target_size=224):
    """
    Preprocess images for the encoder.
    
    Args:
        x: Input images in [0, 1] range, shape (B, C, H, W)
        encoder_type: Type of encoder
        target_size: Target size for encoder input
    
    Returns:
        Preprocessed images ready for encoder
    """
    # Resize to encoder input size
    if x.shape[-1] != target_size:
        x = F.interpolate(x, size=(target_size, target_size), mode='bicubic', align_corners=False)
    
    # Normalize with ImageNet stats
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    
    return x


@torch.no_grad()
def extract_encoder_features(encoder, images, encoder_type="dinov2"):
    """
    Extract features from encoder.
    
    Args:
        encoder: The encoder model
        images: Preprocessed images
        encoder_type: Type of encoder
    
    Returns:
        features: Extracted features (B, num_patches, embed_dim)
    """
    if "dinov2" in encoder_type:
        # DINOv2 returns dict with 'x_norm_patchtokens'
        features = encoder.forward_features(images)
        if isinstance(features, dict):
            features = features['x_norm_patchtokens']
        else:
            # Some versions return the full features, need to extract patch tokens
            features = features[:, 1:]  # Remove CLS token
    else:
        features = encoder.forward_features(images)
    
    return features


def compute_alignment_loss(z_model, z_encoder):
    """
    Compute the alignment loss between model features and encoder features.
    Uses cosine similarity loss as in REPA.
    
    Args:
        z_model: Model intermediate features (B, T, D_model) - should be normalized
        z_encoder: Encoder features (B, T_enc, D_enc) - should be normalized
    
    Returns:
        loss: Negative cosine similarity loss
    """
    # Normalize both
    z_model = F.normalize(z_model, dim=-1)
    z_encoder = F.normalize(z_encoder, dim=-1)
    
    # If sequence lengths don't match, we need to interpolate
    # z_model patches: (input_size / patch_size)^2 = (32/2)^2 = 256 for CIFAR
    # z_encoder patches: (224 / 14)^2 = 256 for DINOv2 with 224 input
    # They should match!
    
    if z_model.shape[1] != z_encoder.shape[1]:
        # Reshape to spatial and interpolate
        B, T_m, D_m = z_model.shape
        B, T_e, D_e = z_encoder.shape
        
        H_m = int(T_m ** 0.5)
        H_e = int(T_e ** 0.5)
        
        z_encoder = z_encoder.reshape(B, H_e, H_e, D_e).permute(0, 3, 1, 2)
        z_encoder = F.interpolate(z_encoder, size=(H_m, H_m), mode='bilinear', align_corners=False)
        z_encoder = z_encoder.permute(0, 2, 3, 1).reshape(B, T_m, D_e)
        z_encoder = F.normalize(z_encoder, dim=-1)
    
    # Cosine similarity loss (negative because we want to maximize similarity)
    loss = mean_flat(-(z_model * z_encoder).sum(dim=-1))
    
    return loss
