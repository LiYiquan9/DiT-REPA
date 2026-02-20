"""
Evaluate linear probe accuracy of DiT features.

Linear probing trains a simple linear classifier on frozen features to measure
how well the learned representations capture semantic information for classification.

Usage:
    # For models trained WITHOUT REPA (uses raw DiT features)
    python evaluate_linear_probe.py --checkpoint checkpoints/20260219_152042/step_30000.pth --num_epochs 100
    
    # For models trained WITH REPA (uses projected DiT features)
    CUDA_VISIBLE_DEVICES=1 python evaluate_linear_probe.py --checkpoint checkpoints/20260219_155341/step_30000.pth --use_repa --num_epochs 100
    
    # Custom options
    python evaluate_linear_probe.py --checkpoint ckpt.pth --use_repa --timestep 0.25 --lr 0.01
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from tqdm import tqdm

from dit import DiT
from ema import LitEma


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


class LinearProbe(nn.Module):
    """Simple linear classifier for probing."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def extract_all_features(model, dataloader, timestep, use_projected, device):
    """Extract features from entire dataset."""
    all_features = []
    all_labels = []
    
    model.eval()
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            features = extract_dit_features(model, images, timestep, use_projected)
        
        all_features.append(features.float().cpu())
        all_labels.append(labels)
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_linear_probe(train_features, train_labels, test_features, test_labels, 
                       num_classes=10, num_epochs=100, lr=0.01, batch_size=256, device="cuda"):
    """Train and evaluate a linear probe."""
    in_dim = train_features.shape[1]
    probe = LinearProbe(in_dim, num_classes).to(device)
    
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        # Evaluation
        probe.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                logits = probe(features)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        best_acc = max(best_acc, test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss/train_total:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
    
    return best_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate linear probe accuracy on DiT features")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights")
    parser.add_argument("--no_ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--use_repa", action="store_true", help="Model was trained with REPA (use projected features)")
    parser.add_argument("--encoder_depth", type=int, default=6, help="DiT layer for feature extraction")
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep for DiT feature extraction")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/cifar10", help="CIFAR-10 data root")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_ema = args.use_ema and not args.no_ema
    
    print(f"{'='*60}")
    print(f"Linear Probe Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Using EMA: {use_ema}")
    print(f"Using REPA model: {args.use_repa}")
    print(f"Timestep: {args.timestep}")
    print(f"Feature extraction layer: {args.encoder_depth}")
    print(f"Training epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # Create DiT model with appropriate config
    if args.use_repa:
        # For REPA models, we need projectors
        # Default to dinov2-s embedding dim
        embed_dim = 384
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
    print(f"\nLoading checkpoint: {args.checkpoint}")
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
    
    # Load datasets
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=T.ToTensor(),
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=T.ToTensor(),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    
    # Extract features
    print(f"\nExtracting features from training set ({len(train_dataset)} samples)...")
    train_features, train_labels = extract_all_features(
        model, train_loader, args.timestep, args.use_repa, device
    )
    
    print(f"Extracting features from test set ({len(test_dataset)} samples)...")
    test_features, test_labels = extract_all_features(
        model, test_loader, args.timestep, args.use_repa, device
    )
    
    print(f"\nTrain features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # Train linear probe
    print(f"\nTraining linear probe for {args.num_epochs} epochs...")
    best_acc, final_acc = train_linear_probe(
        train_features, train_labels,
        test_features, test_labels,
        num_classes=10,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
    
    print(f"\n{'='*60}")
    print(f"Linear Probe Results")
    print(f"{'='*60}")
    print(f"Best Test Accuracy:  {best_acc:.2f}%")
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"{'='*60}")
    
    return {"best_acc": best_acc, "final_acc": final_acc}


if __name__ == "__main__":
    main()
