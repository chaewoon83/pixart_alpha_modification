import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from diffusion.utils.misc import read_config
from diffusion.inference.pixart_infer_withembeddig import build_inference_model, generate_one_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--embed_dir", type=str, required=True, help="Folder containing .npz embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save generated images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--collect_stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    config = read_config(args.config)

    ckpt_path = Path(args.checkpoint).resolve()
    config.work_dir = str(ckpt_path.parent.parent)

    model, vae, t5, latent_size = build_inference_model(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        load_ema=False,
        load_t5=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    log_file = output_dir / "log.txt"
    f_log = open(log_file, "w")

    def log_print(msg):
        print(msg)
        f_log.write(msg + "\n")
        f_log.flush()

    embed_dir = Path(args.embed_dir)
    npz_files = sorted(embed_dir.glob("*.npz"))

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {embed_dir}")

    log_print(f"Config: {args.config}")
    log_print(f"Checkpoint: {args.checkpoint}")
    log_print(f"Device: {args.device}")
    log_print(f"Found {len(npz_files)} embeddings")

    total_times = []

    total_start = time.perf_counter()

    for i, npz_path in enumerate(npz_files):
        log_print(f"[{i+1}/{len(npz_files)}] {npz_path.name}")

        with np.load(npz_path) as data:
            prompt_embeds = torch.from_numpy(data["caption_feature"]).float().to(args.device)
            emb_masks = torch.from_numpy(data["attention_mask"]).to(args.device)

        save_path = output_dir / f"{npz_path.stem}.png"

        
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        generate_one_image(
            config=config,
            model=model,
            vae=vae,
            t5=None,
            latent_size=latent_size,
            prompt_embeds=prompt_embeds,
            emb_masks=emb_masks,
            save_path=str(save_path),
            device=args.device,
            seed=args.seed + i,
            collect_stats=args.collect_stats,
        )

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        total_times.append(elapsed)

        log_print(f"    inference time: {elapsed:.4f} sec")

    total_elapsed = time.perf_counter() - total_start
    avg_time = sum(total_times) / len(total_times)

    log_print("\nDone.")
    log_print(f"Total inference time: {total_elapsed:.4f} sec")
    log_print(f"Average per image:    {avg_time:.4f} sec")
    log_print(f"Min per image:        {min(total_times):.4f} sec")
    log_print(f"Max per image:        {max(total_times):.4f} sec")

    f_log.close()


if __name__ == "__main__":
    main()