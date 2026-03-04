from dit import DiT
import os
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from ema import LitEma
from bitsandbytes.optim import AdamW8bit

from model import RectifiedFlow
from fid_evaluation import FIDEvaluation
from repa import load_encoder, preprocess_for_encoder, extract_encoder_features

import moviepy.editor as mpy
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT with optional REPA alignment")
    # Training
    parser.add_argument("--n_steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # REPA
    parser.add_argument("--use_repa", action="store_true", default=False, help="Enable REPA alignment loss")
    parser.add_argument("--encoder_type", type=str, default="dinov2", help="Encoder type for REPA")
    parser.add_argument("--encoder_size", type=str, default="s", choices=["s", "b", "l", "g"], help="Encoder size")
    parser.add_argument("--proj_coeff", type=float, default=0.5, help="Weight for alignment loss")
    parser.add_argument("--encoder_depth", type=int, default=6, help="Layer at which to extract DiT features for alignment (model has 12 layers)")
    # Model
    parser.add_argument("--dim", type=int, default=384, help="Model hidden dimension")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1, help="Class label dropout probability for CFG")
    # Data
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/cifar10", help="Dataset root directory")
    return parser.parse_args()


def main():
    args = parse_args()
    n_steps = args.n_steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    
    # REPA configuration
    use_repa = args.use_repa
    encoder_type = args.encoder_type
    encoder_size = args.encoder_size
    proj_coeff = args.proj_coeff
    encoder_depth = args.encoder_depth
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"checkpoints/{run_timestamp}"
    img_dir = f"images/{run_timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

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
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    train_dataloader = cycle(train_dataloader)

    # Load encoder for REPA
    encoder = None
    z_dims = None
    if use_repa:
        print(f"Loading {encoder_type}-{encoder_size} encoder for REPA...")
        encoder, embed_dim = load_encoder(encoder_type, encoder_size, device)
        z_dims = [embed_dim]
        print(f"Encoder loaded with embedding dim: {embed_dim}")

    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=10,
        learn_sigma=False,
        class_dropout_prob=args.class_dropout_prob,
        # REPA parameters
        z_dims=z_dims,
        encoder_depth=encoder_depth,
    ).to(device)
    model_ema = LitEma(model)
    optimizer = AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    sampler = RectifiedFlow(model)
    scaler = torch.cuda.amp.GradScaler()

    config = {
        "use_repa": use_repa,
        "encoder_type": encoder_type if use_repa else None,
        "encoder_size": encoder_size if use_repa else None,
        "proj_coeff": proj_coeff if use_repa else None,
        "encoder_depth": encoder_depth if use_repa else None,
        "batch_size": batch_size,
        "n_steps": n_steps,
    }
    logger = wandb.init(project="dit-cfm", config=config)
    fid_eval = FIDEvaluation(batch_size * 2, train_dataloader, sampler)
    
    def sample_and_log_images():
        log_imgs = []
        log_gifs = []
        for cfg_scale in [1.0, 2.5, 5.0]:
            print(
                f"Sampling images at step {step} with cfg_scale {cfg_scale}..."
            )
            final_img, traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            log_img = make_grid(final_img, nrow=10)
            img_save_path = f"{img_dir}/step{step}_cfg{cfg_scale}.png"
            save_image(log_img, img_save_path)
            log_imgs.append(
                wandb.Image(img_save_path, caption=f"cfg_scale: {cfg_scale}")
            )
            # print(f"Saved images to {img_save_path}")
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            clip.write_gif(f"{img_dir}/step{step}_cfg{cfg_scale}.gif")
            log_gifs.append(
                wandb.Video(
                    f"{img_dir}/step{step}_cfg{cfg_scale}.gif",
                    caption=f"cfg_scale: {cfg_scale}",
                )
            )

            print("Copying EMA to model...")
            model_ema.store(model.parameters())
            model_ema.copy_to(model)
            print(
                f"Sampling images with ema model at step {step} with cfg_scale {cfg_scale}..."
            )
            final_img, traj = sampler.sample_each_class(10, cfg_scale=cfg_scale, return_all_steps=True)
            log_img = make_grid(final_img, nrow=10)
            img_save_path = f"{img_dir}/step{step}_cfg{cfg_scale}_ema.png"
            save_image(log_img, img_save_path)
            # print(f"Saved images to {img_save_path}")
            log_imgs.append(
                wandb.Image(
                    img_save_path, caption=f"EMA with cfg_scale: {cfg_scale}"
                )
            )
            
            images_list = [
                make_grid(frame, nrow=10).permute(1, 2, 0).cpu().numpy() * 255
                for frame in traj
            ]
            clip = mpy.ImageSequenceClip(images_list, fps=10)
            clip.write_gif(f"{img_dir}/step{step}_cfg{cfg_scale}_ema.gif")
            log_gifs.append(
                wandb.Video(
                    f"{img_dir}/step{step}_cfg{cfg_scale}_ema.gif",
                    caption=f"EMA with cfg_scale: {cfg_scale}",
                )
            )
            model_ema.restore(model.parameters())
        logger.log({"Images": log_imgs, "Gifs": log_gifs, "step": step})
    
    losses = []
    align_losses = []
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad()
            x1 = data[0].to(device)
            y = data[1].to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if use_repa and encoder is not None:
                    # Extract encoder features
                    with torch.no_grad():
                        x_for_encoder = preprocess_for_encoder(x1, encoder_type, target_size=224)
                        encoder_features = [extract_encoder_features(encoder, x_for_encoder, encoder_type)]
                    
                    # Forward with alignment loss
                    denoising_loss, alignment_loss = sampler(x1, y, encoder_features=encoder_features)
                    loss = denoising_loss + proj_coeff * alignment_loss
                    align_losses.append(alignment_loss.item())
                else:
                    loss = sampler(x1, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema(model)

            if not torch.isnan(loss):
                losses.append(loss.item())
                if use_repa and len(align_losses) > 0:
                    pbar.set_postfix({"loss": loss.item(), "align": align_losses[-1]})
                    logger.log({"loss": loss.item(), "align_loss": align_losses[-1], "step": step})
                else:
                    pbar.set_postfix({"loss": loss.item()})
                    logger.log({"loss": loss.item(), "step": step})

            
            if step % 10000 == 0 or step == n_steps - 1:
                avg_loss = sum(losses) / len(losses) if losses else 0
                avg_align = sum(align_losses) / len(align_losses) if align_losses else 0
                if use_repa:
                    print(f"Step: {step+1}/{n_steps} | loss: {avg_loss:.4f} | align: {avg_align:.4f}")
                else:
                    print(f"Step: {step+1}/{n_steps} | loss: {avg_loss:.4f}")
                losses.clear()
                align_losses.clear()
                model.eval()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sample_and_log_images()
                model.train()
                
                # Save checkpoint
                ckpt = {
                    "model": model.state_dict(),
                    "ema": model_ema.state_dict(),
                    "step": step,
                }
                torch.save(ckpt, f"{ckpt_dir}/step_{step}.pth")
                print(f"Checkpoint saved at step {step}")

            if step % 50000 == 0 or step == n_steps - 1:
                model.eval()
                model_ema.store(model.parameters())
                model_ema.copy_to(model)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fid_score = fid_eval.fid_score()
                print(f"FID score with EMA at step {step}: {fid_score}")
                
                model_ema.restore(model.parameters())
                model.train()
                
                wandb.log({"FID": fid_score, "step": step})

    state_dict = {
        "model": model.state_dict(),
        "ema": model_ema.state_dict(),
    }
    torch.save(state_dict, "model.pth")


if __name__ == "__main__":
    main()
