"""
Evaluate FID score from a pretrained checkpoint.

Usage:
    python evaluate_fid.py --checkpoint checkpoints/20260219_152042/step_30000.pth
    python evaluate_fid.py --checkpoint checkpoints/20260219_155341/step_30000.pth --use_repa
"""
import argparse
import torch
import torchvision
from torchvision import transforms as T

from dit import DiT
from ema import LitEma
from model import RectifiedFlow
from fid_evaluation import FIDEvaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights (default: True)")
    parser.add_argument("--no_ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--use_repa", action="store_true", help="Model was trained with REPA (has projectors)")
    parser.add_argument("--encoder_depth", type=int, default=6, help="Encoder depth for REPA model")
    parser.add_argument("--z_dim", type=int, default=384, help="Encoder embedding dim for REPA (384 for DINOv2-S)")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale for sampling")
    parser.add_argument("--sample_steps", type=int, default=25, help="Number of sampling steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for FID evaluation")
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/cifar10", help="CIFAR-10 data root")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine if using EMA
    use_ema = args.use_ema and not args.no_ema
    
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Using EMA: {use_ema}")
    print(f"REPA model: {args.use_repa}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Sample steps: {args.sample_steps}")
    
    # Load dataset
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=T.Compose([T.ToTensor(), T.RandomHorizontalFlip()]),
    )
    
    def cycle(iterable):
        while True:
            for i in iterable:
                yield i
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    train_dataloader = cycle(train_dataloader)
    
    # Create model with appropriate config
    if args.use_repa:
        z_dims = [args.z_dim]
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
            # REPA parameters
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
        ).to(device)
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
        ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if use_ema and "ema" in ckpt:
        # Load EMA weights
        model_ema = LitEma(model)
        model_ema.load_state_dict(ckpt["ema"])
        model_ema.copy_to(model)
        print("Loaded EMA weights")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    else:
        # Try loading directly (old format)
        model.load_state_dict(ckpt)
        print("Loaded weights directly from checkpoint")
    
    if "step" in ckpt:
        print(f"Checkpoint step: {ckpt['step']}")
    
    model.eval()
    
    # Create sampler and FID evaluator
    sampler = RectifiedFlow(model)
    fid_eval = FIDEvaluation(args.batch_size, train_dataloader, sampler)
    
    # Evaluate FID
    print(f"\nEvaluating FID with cfg_scale={args.cfg_scale}, sample_steps={args.sample_steps}...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fid_score = fid_eval.fid_score(cfg_scale=args.cfg_scale, sample_steps=args.sample_steps)
    
    print(f"\n{'='*50}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"{'='*50}")
    
    return fid_score


if __name__ == "__main__":
    main()
