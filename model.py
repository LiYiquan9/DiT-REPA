import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from tqdm.auto import tqdm

from dit import DiT


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

class RectifiedFlow(nn.Module):
    def __init__(
        self,
        net: DiT,
        device="cuda",
        channels=3,
        image_size=32,
        num_classes=10,
        logit_normal_sampling_t=True,
    ):
        super().__init__()
        self.net = net
        self.device = device
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.logit_normal_sampling_t = logit_normal_sampling_t

    def forward(self, x, c=None, encoder_features=None):
        """
        Forward pass for training.
        
        Args:
            x: Input images (B, C, H, W) in [0, 1] range
            c: Class labels (B,)
            encoder_features: Optional list of encoder features for REPA alignment
        
        Returns:
            If encoder_features is None: denoising loss
            If encoder_features provided: (denoising_loss, alignment_loss)
        """
        if self.logit_normal_sampling_t:
            t = torch.randn((x.shape[0],), device=self.device).sigmoid()
        else:
            t = torch.rand((x.shape[0],), device=self.device)
        
        t_ = rearrange(t, "b -> b 1 1 1")
        z = torch.randn_like(x)
        x = normalize_to_neg1_1(x)
        z_t = (1 - t_) * x + t_ * z
        target = z - x
        
        # Check if we need to compute alignment loss
        if encoder_features is not None:
            v_t, zs_model, _ = self.net(z_t, t, c, return_features=True)
            denoising_loss = F.mse_loss(target, v_t)
            
            # Compute alignment loss
            alignment_loss = 0.0
            if zs_model is not None:
                for z_model, z_enc in zip(zs_model, encoder_features):
                    alignment_loss = alignment_loss + self._compute_alignment_loss(z_model, z_enc)
                alignment_loss = alignment_loss / len(encoder_features)
            
            return denoising_loss, alignment_loss
        else:
            v_t = self.net(z_t, t, c)
            return F.mse_loss(target, v_t)
    
    def _compute_alignment_loss(self, z_model, z_encoder):
        """Compute cosine similarity alignment loss."""
        # Normalize both
        z_model = F.normalize(z_model, dim=-1)
        z_encoder = F.normalize(z_encoder, dim=-1)
        
        # Handle sequence length mismatch
        if z_model.shape[1] != z_encoder.shape[1]:
            B, T_m, D_m = z_model.shape
            B, T_e, D_e = z_encoder.shape
            
            H_m = int(T_m ** 0.5)
            H_e = int(T_e ** 0.5)
            
            z_encoder = z_encoder.reshape(B, H_e, H_e, D_e).permute(0, 3, 1, 2)
            z_encoder = F.interpolate(z_encoder, size=(H_m, H_m), mode='bilinear', align_corners=False)
            z_encoder = z_encoder.permute(0, 2, 3, 1).reshape(B, T_m, D_e)
            z_encoder = F.normalize(z_encoder, dim=-1)
        
        # Negative cosine similarity (we want to maximize similarity)
        loss = -(z_model * z_encoder).sum(dim=-1).mean()
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        if self.use_cond:
            y = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        
        z = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)
        
        images = [z]
        t_span = torch.linspace(0, 1, sample_steps, device=self.device)
        for t in tqdm(reversed(t_span)):
            if self.use_cond:
                v_t = self.net.forward_with_cfg(z, t, y, cfg_scale)
            else:
                v_t = self.net(z, t)
            z = z - v_t / sample_steps
            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)
        return z
        
    
    @torch.no_grad()
    def sample_each_class(self, n_per_class, cfg_scale=5.0, sample_steps=50, return_all_steps=False):
        c = torch.arange(self.num_classes, device=self.device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels, self.image_size, self.image_size, device=self.device)
        
        images = []
        t_span = torch.linspace(0, 1, sample_steps, device=self.device)
        for t in tqdm(reversed(t_span)):
            v_t = self.net.forward_with_cfg(z, t, c, cfg_scale)
            z = z - v_t / sample_steps
            images.append(z)
        
        z = unnormalize_to_0_1(z.clip(-1, 1))
        
        if return_all_steps:
            return z, torch.stack(images)
        return z