import argparse
import sys
import time
from pathlib import Path
import torch

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from diffusion.utils.misc import read_config
from diffusion.inference.pixart_infer import build_inference_model, generate_one_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
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
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    log_file = output_dir / "log.txt"
    f_log = open(log_file, "w")

    def log_print(msg):
        print(msg)
        f_log.write(msg + "\n")
        f_log.flush()

    log_print(f"Config: {args.config}")
    log_print(f"Checkpoint: {args.checkpoint}")
    log_print(f"Prompt: {args.prompt}")
    log_print(f"Device: {args.device}")
    log_print(f"Output dir: {args.output_dir}")

    total_times = []

    total_start = time.perf_counter()

    # Generate single image with prompt
    log_print(f"[1/1] Processing prompt: {args.prompt}")

    save_path = output_dir / f"output_{args.seed}.png"

    
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generate_one_image(
        config=config,
        model=model,
        vae=vae,
        t5=t5,
        latent_size=latent_size,
        prompt=args.prompt,
        save_path=str(save_path),
        device=args.device,
        seed=args.seed,
        collect_stats=args.collect_stats,
    )

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    total_times.append(inference_time)

    log_print(f"[1/1] Completed in {inference_time:.3f}s - saved to {save_path}")

    total_end = time.perf_counter()
    total_time = total_end - total_start

    log_print(f"Total time: {total_time:.3f}s")
    log_print(f"Average inference time: {sum(total_times)/len(total_times):.3f}s")
    log_print(f"Images generated: {len(total_times)}")

    f_log.close()


if __name__ == "__main__":
    main()