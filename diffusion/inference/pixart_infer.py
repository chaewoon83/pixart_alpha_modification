import os
import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM, DPMS, SASolverSampler


def build_inference_model(config, checkpoint_path, device="cuda", load_ema=False):
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

    print("[5] loading T5...")
    t5 = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir="output/pretrained_models/t5_ckpts",
        torch_dtype=torch.float,
    )

    print("[6] build finished")
    return model, vae, t5, latent_size


@torch.no_grad()
def generate_one_image(
    config,
    model,
    vae,
    t5,
    latent_size,
    prompt,
    save_path="output.png",
    device="cuda",
    seed=0,
):
    print("[7] start generation")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()

    print("[8] text embedding...")
    caption_embs, emb_masks = t5.get_text_embeddings([prompt])
    caption_embs = caption_embs.float()[:, None]
    print("[8-1] caption_embs shape:", caption_embs.shape)

    null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]
    print("[8-2] null_y shape:", null_y.shape)

    hw = torch.tensor([[config.image_size, config.image_size]], dtype=torch.float, device=device)
    ar = torch.tensor([[1.0]], dtype=torch.float, device=device)

    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    print("[8-3] initial noise stats:",
          "min=", z.min().item(),
          "max=", z.max().item(),
          "mean=", z.mean().item())

    
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

    print("[11] latent stats before decode:",
          "min=", samples.min().item(),
          "max=", samples.max().item(),
          "mean=", samples.mean().item())

    print("[12] decoding with VAE...")
    samples = vae.decode(samples / 0.18215).sample

    img = (samples.clamp(-1, 1) + 1) / 2

    print("[13] image stats after decode:",
          "min=", img.min().item(),
          "max=", img.max().item(),
          "mean=", img.mean().item())

    save_image(img, save_path)
    print(f"[14] Saved image to {save_path}")