import os
import json
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM, DPMS, SASolverSampler


def build_inference_model(config, checkpoint_path, device="cuda", load_ema=False, load_t5=True):
    os.makedirs(config.work_dir, exist_ok=True)

    print("[1] start build_inference_model")
    image_size = config.image_size
    latent_size = image_size // 8

    pred_sigma = getattr(config, "pred_sigma", True)
    learn_sigma = getattr(config, "learn_sigma", True) and pred_sigma

    model_kwargs = {
        "window_block_indexes": config.window_block_indexes,
        "window_size": config.window_size,
        "use_rel_pos": config.use_rel_pos,
        "lewei_scale": config.lewei_scale,
        "config": config,
        "model_max_length": config.model_max_length,
    }

    print("[2] building model...")
    model = build_model(
        config.model,
        config.grad_checkpointing,
        config.get("fp32_attention", False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **model_kwargs
    ).eval().to(device)

    print("[3] loading checkpoint...")
    missing, unexpected = load_checkpoint(checkpoint_path, model, load_ema=load_ema)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    print("[4] loading VAE...")
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(device)

    t5 = None
    if load_t5:
        print("[5] loading T5...")
        t5 = T5Embedder(
            device=device,
            local_cache=True,
            cache_dir="output/pretrained_models/t5_ckpts",
            torch_dtype=torch.float,
        )
    else:
        print("[5] skipping T5 loading (using precomputed embeddings)")

    print("[6] build finished")
    return model, vae, t5, latent_size


def _get_target_model(model):
    return model.module if hasattr(model, "module") else model


def _is_routed_mlp(mlp):
    return (
        hasattr(mlp, "reset_stats")
        and hasattr(mlp, "total_tokens")
        and hasattr(mlp, "total_light_tokens")
        and hasattr(mlp, "total_heavy_tokens")
        and hasattr(mlp, "last_routing_map")
        and hasattr(mlp, "num_forwards")
    )


@torch.no_grad()
def generate_one_image(
    config,
    model,
    vae,
    t5,
    latent_size,
    prompt=None,
    prompt_embeds=None,
    emb_masks=None,
    save_path="output.png",
    device="cuda",
    seed=0,
    collect_stats=False,
):
    print("[7] start generation")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    target_model = _get_target_model(model)

    if collect_stats:
        for block in target_model.blocks:
            if hasattr(block, "mlp") and _is_routed_mlp(block.mlp):
                block.mlp.reset_stats()

    print("[8] prepare text condition...")
    if prompt_embeds is None:
        if prompt is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")
        if t5 is None:
            raise ValueError("t5 is None, so prompt_embeds/emb_masks must be provided.")

        caption_embs, emb_masks = t5.get_text_embeddings([prompt])
        caption_embs = caption_embs.float()[:, None]
        emb_masks = emb_masks.to(device)
    else:
        caption_embs = prompt_embeds.to(device).float()
        if caption_embs.ndim == 3:
            caption_embs = caption_embs[:, None]
        elif caption_embs.ndim != 4:
            raise ValueError(f"prompt_embeds must have 3 or 4 dims, got shape {caption_embs.shape}")

        if emb_masks is None:
            raise ValueError("emb_masks must be provided with prompt_embeds.")
        emb_masks = emb_masks.to(device)

    print("[8-1] caption_embs shape:", caption_embs.shape)
    print("[8-1] emb_masks shape:", emb_masks.shape)

    null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None].to(device)
    print("[8-2] null_y shape:", null_y.shape)

    hw = torch.tensor([[config.image_size, config.image_size]], dtype=torch.float, device=device)
    ar = torch.tensor([[1.0]], dtype=torch.float, device=device)

    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    print(
        "[8-3] initial noise stats:",
        "min=", z.min().item(),
        "max=", z.max().item(),
        "mean=", z.mean().item()
    )

    sampler_name = "iddpm"
    eval_steps = 100
    cfg_scale = 1.0

    print("[9] Using sampler:", sampler_name)
    print("[9] eval_steps:", eval_steps)
    print("[9] cfg_scale:", cfg_scale)

    model_kwargs = dict(
        data_info={"img_hw": hw, "aspect_ratio": ar},
        mask=emb_masks,
    )

    if sampler_name == "iddpm":
        sample_steps = eval_steps
        z = z.repeat(2, 1, 1, 1)

        model_kwargs = dict(
            y=torch.cat([caption_embs, null_y]),
            cfg_scale=cfg_scale,
            data_info={"img_hw": hw, "aspect_ratio": ar},
            mask=emb_masks,
        )

        print("[10] start IDDPM sampling...")
        diffusion = IDDPM(str(sample_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
        samples, _ = samples.chunk(2, dim=0)

    elif sampler_name == "dpm-solver":
        sample_steps = eval_steps

        print("[10] start DPM-Solver sampling...")
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_scale,
            model_kwargs=model_kwargs,
        )

        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

    elif sampler_name == "sa-solver":
        sample_steps = eval_steps

        print("[10] start SA-Solver sampling...")
        sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sa_solver.sample(
            S=sample_steps,
            batch_size=1,
            shape=(4, latent_size, latent_size),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=cfg_scale,
            model_kwargs=model_kwargs,
        )[0]

    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Save routing log only when collect_stats=True
    if collect_stats:
        grand_light = 0
        grand_heavy = 0
        grand_total = 0
        block_logs = []

        for i, block in enumerate(target_model.blocks):
            if not hasattr(block, "mlp"):
                continue
            if not _is_routed_mlp(block.mlp):
                continue

            mlp = block.mlp

            if mlp.total_tokens == 0:
                continue

            image_light_ratio = mlp.total_light_tokens / mlp.total_tokens
            image_heavy_ratio = mlp.total_heavy_tokens / mlp.total_tokens

            grand_light += mlp.total_light_tokens
            grand_heavy += mlp.total_heavy_tokens
            grand_total += mlp.total_tokens

            token_routing = None
            if mlp.last_routing_map is not None:
                token_routing = mlp.last_routing_map.reshape(-1).tolist()

            block_logs.append({
                "block_index": i,
                "token_last_step_light_vs_heavy": token_routing,  # 0=light, 1=heavy
                "image_light_tokens": int(mlp.total_light_tokens),
                "image_heavy_tokens": int(mlp.total_heavy_tokens),
                "image_total_tokens": int(mlp.total_tokens),
                "image_light_ratio": float(image_light_ratio),
                "image_heavy_ratio": float(image_heavy_ratio),
                "num_forwards": int(mlp.num_forwards),
            })

        total_log = {
            "save_path": str(save_path),
            "prompt": prompt,
            "seed": int(seed),
            "sampler": sampler_name,
            "eval_steps": int(eval_steps),
            "cfg_scale": float(cfg_scale),
            "total_light_tokens": int(grand_light),
            "total_heavy_tokens": int(grand_heavy),
            "total_tokens": int(grand_total),
            "total_light_ratio": float(grand_light / grand_total) if grand_total > 0 else None,
            "total_heavy_ratio": float(grand_heavy / grand_total) if grand_total > 0 else None,
            "blocks": block_logs,
        }

        log_path = Path(save_path).with_suffix(".routing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(total_log, f, indent=2)

        print(f"[10-1] Saved routing stats to {log_path}")

    print(
        "[11] latent stats before decode:",
        "min=", samples.min().item(),
        "max=", samples.max().item(),
        "mean=", samples.mean().item()
    )

    print("[12] decoding with VAE...")
    samples = vae.decode(samples / 0.18215).sample

    img = (samples.clamp(-1, 1) + 1) / 2

    print(
        "[13] image stats after decode:",
        "min=", img.min().item(),
        "max=", img.max().item(),
        "mean=", img.mean().item()
    )

    save_image(img, save_path)
    print(f"[14] Saved image to {save_path}")