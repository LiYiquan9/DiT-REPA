"""
Evaluate CKNNA (Centered Kernel Nearest-Neighbor Alignment) between DiT and encoder features.

CKNNA measures how well the DiT's intermediate representations align with a pretrained encoder
(e.g., DINOv2). Higher CKNNA indicates better representation alignment.

Usage:
    # For models trained WITHOUT REPA (uses raw DiT features)
    python evaluate_cknna.py --checkpoint checkpoints/20260219_152042/step_199999.pth --num_samples 20000
    
    # For models trained WITH REPA (uses projected DiT features)
    CUDA_VISIBLE_DEVICES=1 python evaluate_cknna.py --checkpoint checkpoints/20260219_155341/step_199999.pth --use_repa --num_samples 20000
    
    # Custom options
    python evaluate_cknna.py --checkpoint ckpt.pth --use_repa --topk 10 --encoder_depth 6
"""
import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from tqdm import tqdm

from dit import DiT
from ema import LitEma
from repa import load_encoder, preprocess_for_encoder, extract_encoder_features


def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC).
    Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """Compute the biased HSIC (the original CKA)"""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cknna(feats_A, feats_B, topk=10, unbiased=True):
    """
    Compute CKNNA (Centered Kernel Nearest-Neighbor Alignment).
    
    CKNNA is a relaxed version of CKA that only considers k-nearest neighbors,
    making it more robust to outliers and more sensitive to local structure.
    
    Args:
        feats_A: Features from model A, shape (N, D1)
        feats_B: Features from model B, shape (N, D2)
        topk: Number of nearest neighbors to consider (default: 10)
        unbiased: Whether to use unbiased HSIC estimator (default: True)
    
    Returns:
        CKNNA score (float between 0 and 1, higher is better alignment)
    """
    n = feats_A.shape[0]
    device = feats_A.device
    
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    
    # Normalize features
    feats_A = F.normalize(feats_A, dim=-1)
    feats_B = F.normalize(feats_B, dim=-1)
    
    # Compute kernel matrices (inner product kernel)
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T

    def compute_similarity(K, L, topk):
        if unbiased:
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        # Get topk indices for each row
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
        
        # Create masks for nearest neighbors
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
        
        # Intersection of nearest neighbors (Eq. 24 in REPA paper)
        mask = mask_K * mask_L
        
        # Compute HSIC on masked kernel matrices
        if unbiased:
            sim = hsic_unbiased(mask * K, mask * L)
        else:
            sim = hsic_biased(mask * K, mask * L)
        return sim

    sim_kl = compute_similarity(K, L, topk)
    sim_kk = compute_similarity(K, K, topk)
    sim_ll = compute_similarity(L, L, topk)
    
    # CKNNA formula (Eq. 21 with relaxed alignment)
    cknna_score = sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()
    
    return cknna_score


def cka(feats_A, feats_B, unbiased=True):
    """
    Compute CKA (Centered Kernel Alignment) - the non-relaxed version.
    
    Args:
        feats_A: Features from model A, shape (N, D1)
        feats_B: Features from model B, shape (N, D2)
        unbiased: Whether to use unbiased HSIC estimator
    
    Returns:
        CKA score (float between 0 and 1)
    """
    # Normalize features
    feats_A = F.normalize(feats_A, dim=-1)
    feats_B = F.normalize(feats_B, dim=-1)
    
    # Compute kernel matrices
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    
    # Compute HSIC values
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    hsic_kl = hsic_fn(K, L)
    
    # CKA formula
    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return cka_value.item()


def mutual_knn(feats_A, feats_B, topk=10):
    """
    Compute mutual KNN accuracy.
    Measures what fraction of k-nearest neighbors are shared between two feature spaces.
    """
    feats_A = F.normalize(feats_A, dim=-1)
    feats_B = F.normalize(feats_B, dim=-1)
    
    n = feats_A.shape[0]
    device = feats_A.device
    
    # Compute similarity matrices
    sim_A = feats_A @ feats_A.T
    sim_B = feats_B @ feats_B.T
    
    # Get KNN indices (excluding self)
    sim_A.fill_diagonal_(float("-inf"))
    sim_B.fill_diagonal_(float("-inf"))
    
    _, knn_A = torch.topk(sim_A, topk, dim=1)
    _, knn_B = torch.topk(sim_B, topk, dim=1)
    
    # Create binary masks
    range_tensor = torch.arange(n, device=device).unsqueeze(1)
    mask_A = torch.zeros(n, n, device=device)
    mask_B = torch.zeros(n, n, device=device)
    mask_A[range_tensor, knn_A] = 1.0
    mask_B[range_tensor, knn_B] = 1.0
    
    # Mutual KNN accuracy
    acc = (mask_A * mask_B).sum(dim=1) / topk
    return acc.mean().item()


@torch.no_grad()
def extract_dit_features(model, images, timestep=0.5, use_projected=True):
    """
    Extract intermediate features from DiT model.
    
    Args:
        model: DiT model
        images: Input images (B, C, H, W) in [0, 1] range
        timestep: Diffusion timestep to use for feature extraction
        use_projected: If True, use projected features (REPA model), else use raw features
    
    Returns:
        features: DiT features averaged over patches (B, dim)
    """
    device = next(model.parameters()).device
    B = images.shape[0]
    
    # Normalize images to [-1, 1]
    x = images * 2 - 1
    
    # Create noised input at specified timestep
    t = torch.full((B,), timestep, device=device)
    noise = torch.randn_like(x)
    
    # For rectified flow: z_t = (1-t)*x + t*noise
    t_expanded = t.view(-1, 1, 1, 1)
    z_t = (1 - t_expanded) * x + t_expanded * noise
    
    # Random class labels
    y = torch.randint(0, model.num_classes, (B,), device=device)
    
    # Forward pass with feature extraction
    _, zs, raw_features = model(z_t, t, y, return_features=True)
    
    if use_projected:
        if zs is None or len(zs) == 0:
            raise ValueError("Model does not have projectors. Use --use_repa for REPA-trained models or don't use --use_projected.")
        # Return first projector's features, averaged over patches
        features = zs[0].mean(dim=1)
    else:
        if raw_features is None:
            raise ValueError("Failed to extract raw features from DiT.")
        # Return raw features, averaged over patches
        features = raw_features.mean(dim=1)
    
    return features


def main():
    parser = argparse.ArgumentParser(description="Evaluate CKNNA alignment score")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights")
    parser.add_argument("--no_ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--use_repa", action="store_true", help="Model was trained with REPA")
    parser.add_argument("--encoder_type", type=str, default="dinov2", help="Encoder type")
    parser.add_argument("--encoder_size", type=str, default="s", help="Encoder size (s/b/l/g)")
    parser.add_argument("--encoder_depth", type=int, default=6, help="DiT layer for feature extraction")
    parser.add_argument("--topk", type=int, default=10, help="K for CKNNA nearest neighbors")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for evaluation")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction")
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep for DiT feature extraction")
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/cifar10", help="CIFAR-10 data root")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_ema = args.use_ema and not args.no_ema
    
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Using EMA: {use_ema}")
    print(f"Using REPA model: {args.use_repa}")
    print(f"Encoder: {args.encoder_type}-{args.encoder_size}")
    print(f"CKNNA topk: {args.topk}")
    print(f"Num samples: {args.num_samples}")
    print(f"Timestep: {args.timestep}")
    print(f"Feature extraction layer: {args.encoder_depth}")
    
    # Load encoder
    print(f"\nLoading {args.encoder_type}-{args.encoder_size} encoder...")
    encoder, embed_dim = load_encoder(args.encoder_type, args.encoder_size, device)
    print(f"Encoder embedding dim: {embed_dim}")
    
    # Create DiT model with appropriate config
    if args.use_repa:
        z_dims = [embed_dim]
        model = DiT(
            input_size=32,
            patch_size=2,
            in_channels=3,
            dim=384,
            depth=12,
            num_heads=6,
            num_classes=10,
            learn_sigma=False,
            class_dropout_prob=0.1,
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
        ).to(device)
        print("Created DiT model with REPA projectors")
    else:
        model = DiT(
            input_size=32,
            patch_size=2,
            in_channels=3,
            dim=384,
            depth=12,
            num_heads=6,
            num_classes=10,
            learn_sigma=False,
            class_dropout_prob=0.1,
            encoder_depth=args.encoder_depth,  # Still need this for raw feature extraction
        ).to(device)
        print("Created DiT model without REPA projectors (will use raw features)")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if use_ema and "ema" in ckpt:
        model_ema = LitEma(model)
        model_ema.load_state_dict(ckpt["ema"])
        model_ema.copy_to(model)
        print("Loaded EMA weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    else:
        model.load_state_dict(ckpt)
        print("Loaded weights directly")
    
    if "step" in ckpt:
        print(f"Checkpoint step: {ckpt['step']}")
    
    model.eval()
    encoder.eval()
    
    # Load dataset
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=T.ToTensor(),
    )
    
    # Limit to num_samples
    if args.num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    
    print(f"\nExtracting features from {len(dataset)} samples...")
    
    all_dit_features = []
    all_encoder_features = []
    
    for images, _ in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Extract DiT features (projected if REPA, raw if not)
            dit_feats = extract_dit_features(model, images, args.timestep, use_projected=args.use_repa)
            
            # Extract encoder features
            x_enc = preprocess_for_encoder(images, args.encoder_type, target_size=224)
            enc_feats = extract_encoder_features(encoder, x_enc, args.encoder_type)
            # Average over patches: (B, num_patches, embed_dim) -> (B, embed_dim)
            enc_feats = enc_feats.mean(dim=1)
        
        all_dit_features.append(dit_feats.float().cpu())
        all_encoder_features.append(enc_feats.float().cpu())
    
    # Concatenate all features
    all_dit_features = torch.cat(all_dit_features, dim=0)
    all_encoder_features = torch.cat(all_encoder_features, dim=0)
    
    print(f"\nDiT features shape: {all_dit_features.shape}")
    print(f"Encoder features shape: {all_encoder_features.shape}")
    
    # Move to GPU for faster computation
    all_dit_features = all_dit_features.to(device)
    all_encoder_features = all_encoder_features.to(device)
    
    # Compute metrics
    print(f"\nComputing alignment metrics...")
    
    cknna_score = cknna(all_dit_features, all_encoder_features, topk=args.topk)
    cka_score = cka(all_dit_features, all_encoder_features)
    mknn_score = mutual_knn(all_dit_features, all_encoder_features, topk=args.topk)
    
    print(f"\n{'='*60}")
    print(f"Alignment Metrics (DiT vs {args.encoder_type}-{args.encoder_size})")
    print(f"{'='*60}")
    print(f"CKNNA (k={args.topk}):      {cknna_score:.4f}")
    print(f"CKA:                 {cka_score:.4f}")
    print(f"Mutual KNN (k={args.topk}): {mknn_score:.4f}")
    print(f"{'='*60}")
    
    return {
        "cknna": cknna_score,
        "cka": cka_score,
        "mutual_knn": mknn_score,
    }


if __name__ == "__main__":
    main()
