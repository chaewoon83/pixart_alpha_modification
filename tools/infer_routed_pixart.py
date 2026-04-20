import argparse
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from diffusion.utils.misc import read_config
from diffusion.inference.pixart_infer import build_inference_model, generate_one_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--save_path", type=str, default="output.png", help="Output image path")
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

    generate_one_image(
        config=config,
        model=model,
        vae=vae,
        t5=t5,
        latent_size=latent_size,
        prompt=args.prompt,
        save_path=args.save_path,
        device=args.device,
        seed=args.seed,
        collect_stats=args.collect_stats,
    )


if __name__ == "__main__":
    main()