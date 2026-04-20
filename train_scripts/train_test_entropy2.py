import argparse
import datetime
import os
import sys
import time
import types
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.model.nets.routed_ffn import RoutedFFN

warnings.filterwarnings("ignore")

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def load_pretrained_to_routed(model, ckpt_path, logger=None):
    """
    Load original PixArt pretrained checkpoint into RoutedFFN model.
    Mapping:
      blocks.i.mlp.fc1.* -> blocks.i.mlp.heavy.fc1.*
      blocks.i.mlp.fc2.* -> blocks.i.mlp.heavy.fc2.*
    Other matching keys are loaded as-is.
    """
    print(f"Loading pretrained checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()

    new_state = {}
    skipped = []

    for k, v in state.items():
        if ".mlp.fc1." in k:
            new_k = k.replace(".mlp.fc1.", ".mlp.heavy.fc1.")
        elif ".mlp.fc2." in k:
            new_k = k.replace(".mlp.fc2.", ".mlp.heavy.fc2.")
        else:
            new_k = k

        if new_k in model_state and model_state[new_k].shape == v.shape:
            new_state[new_k] = v
        else:
            skipped.append((k, new_k, tuple(v.shape)))

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    msg1 = f"[Pretrained load] Loaded keys: {len(new_state)}"
    msg2 = f"[Pretrained load] Missing keys count: {len(missing)}"
    msg3 = f"[Pretrained load] Unexpected keys count: {len(unexpected)}"
    msg4 = f"[Pretrained load] Skipped keys count: {len(skipped)}"

    if logger is not None:
        logger.warning(msg1)
        logger.warning(msg2)
        logger.warning(msg3)
        logger.warning(msg4)

        if len(missing) > 0:
            logger.warning(f"[Pretrained load] Missing keys (first 50): {missing[:50]}")
        if len(unexpected) > 0:
            logger.warning(f"[Pretrained load] Unexpected keys (first 50): {unexpected[:50]}")
        if len(skipped) > 0:
            logger.warning(f"[Pretrained load] Skipped sample (first 20): {skipped[:20]}")
    else:
        print(msg1)
        print(msg2)
        print(msg3)
        print(msg4)
        if len(missing) > 0:
            print("[Pretrained load] Missing keys (first 50):", missing[:50])
        if len(unexpected) > 0:
            print("[Pretrained load] Unexpected keys (first 50):", unexpected[:50])
        if len(skipped) > 0:
            print("[Pretrained load] Skipped sample (first 20):", skipped[:20])

    return missing, unexpected, skipped


def train():
    lambda_router = 0.0001
    lambda_router_entropy = 0.0001

    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')

    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0

        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start

            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()

            clean_images = z * config.scale_factor
            y = batch[1]
            y_mask = batch[2]
            data_info = batch[3]

            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                loss_term = train_diffusion.training_losses(
                    model,
                    clean_images,
                    timesteps,
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                )
                diffusion_loss = loss_term["loss"].mean()

                model_unwrapped = accelerator.unwrap_model(model)
                routed_modules = [m for m in model_unwrapped.modules() if isinstance(m, RoutedFFN)]

                router_loss_list = []
                entropy_list = []
                light_ratio_list = []
                heavy_ratio_list = []
                confidence_list = []
                hard_light_ratio_list = []
                hard_heavy_ratio_list = []

                for m in routed_modules:
                    if m.last_router_loss is not None:
                        router_loss_list.append(m.last_router_loss)
                    if m.last_entropy is not None:
                        entropy_list.append(m.last_entropy)
                    if m.last_light_ratio is not None:
                        light_ratio_list.append(m.last_light_ratio)
                    if m.last_heavy_ratio is not None:
                        heavy_ratio_list.append(m.last_heavy_ratio)
                    if m.last_confidence is not None:
                        confidence_list.append(m.last_confidence)
                    if m.last_hard_light_ratio is not None:
                        hard_light_ratio_list.append(m.last_hard_light_ratio)
                    if m.last_hard_heavy_ratio is not None:
                        hard_heavy_ratio_list.append(m.last_hard_heavy_ratio)

                router_loss = torch.stack(router_loss_list).mean() if len(router_loss_list) > 0 else torch.tensor(0.0, device=clean_images.device)
                routing_entropy = torch.stack(entropy_list).mean() if len(entropy_list) > 0 else torch.tensor(0.0, device=clean_images.device)
                light_ratio = torch.stack(light_ratio_list).mean() if len(light_ratio_list) > 0 else torch.tensor(0.0, device=clean_images.device)
                heavy_ratio = torch.stack(heavy_ratio_list).mean() if len(heavy_ratio_list) > 0 else torch.tensor(0.0, device=clean_images.device)
                routing_confidence = torch.stack(confidence_list).mean() if len(confidence_list) > 0 else torch.tensor(0.0, device=clean_images.device)
                hard_light_ratio = torch.stack(hard_light_ratio_list).mean() if len(hard_light_ratio_list) > 0 else torch.tensor(0.0, device=clean_images.device)
                hard_heavy_ratio = torch.stack(hard_heavy_ratio_list).mean() if len(hard_heavy_ratio_list) > 0 else torch.tensor(0.0, device=clean_images.device)

                loss = diffusion_loss + lambda_router * router_loss + lambda_router_entropy * routing_entropy

                if global_step < 5:
                    print(f"[Step {global_step}] diffusion_loss:", diffusion_loss.item())
                    print(f"[Step {global_step}] router_loss:", router_loss.item())
                    print(f"[Step {global_step}] routing_entropy:", routing_entropy.item())
                    print(f"[Step {global_step}] light_ratio:", light_ratio.item())
                    print(f"[Step {global_step}] heavy_ratio:", heavy_ratio.item())
                    print(f"[Step {global_step}] hard_light_ratio:", hard_light_ratio.item())
                    print(f"[Step {global_step}] hard_heavy_ratio:", hard_heavy_ratio.item())
                    print(f"[Step {global_step}] routing_confidence:", routing_confidence.item())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)

                optimizer.step()
                lr_scheduler.step()

                if accelerator.sync_gradients:
                    ema_update(model_ema, model, config.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]

            logs = {
                "loss": loss.item(),
                "diffusion_loss": diffusion_loss.item(),
                "router_loss": router_loss.item(),
                "routing_entropy": routing_entropy.item(),
                "light_ratio": light_ratio.item(),
                "heavy_ratio": heavy_ratio.item(),
                "hard_light_ratio": hard_light_ratio.item(),
                "hard_heavy_ratio": hard_heavy_ratio.item(),
                "routing_confidence": routing_confidence.item(),
            }

            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())

            log_buffer.update(logs)

            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                model_unwrapped = accelerator.unwrap_model(model)
                log_buffer.average()
                info = (
                    f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}]"
                    f"[{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, "
                    f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, "

                    f"lr:{lr:.3e}, s:({model_unwrapped.h}, {model_unwrapped.w}), "
                )
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)

                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0

            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            global_step += 1
            data_time_start = time.time()

            if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(
                        os.path.join(config.work_dir, 'checkpoints'),
                        epoch=epoch,
                        step=(epoch - 1) * len(train_dataloader) + step + 1,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler
                    )

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(
                    os.path.join(config.work_dir, 'checkpoints'),
                    epoch=epoch,
                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                    model=accelerator.unwrap_model(model),
                    model_ema=accelerator.unwrap_model(model_ema),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)

    if args.work_dir is not None:
        config.work_dir = args.work_dir

    if args.cloud:
        config.data_root = '/data/data'

    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True
        )

    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 8
        config.valid_num = 100

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)

    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches = False

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")

    image_size = config.image_size
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs = {
        "window_block_indexes": config.window_block_indexes,
        "window_size": config.window_size,
        "use_rel_pos": config.use_rel_pos,
        "lewei_scale": config.lewei_scale,
        'config': config,
        'model_max_length': config.model_max_length
    }

    # build models
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss
    )

    model = build_model(
        config.model,
        config.grad_checkpointing,
        config.get('fp32_attention', False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **model_kwargs
    ).train()

    num_routed = sum(1 for m in model.modules() if isinstance(m, RoutedFFN))
    print("Number of RoutedFFN modules:", num_routed)
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # load pretrained BEFORE creating EMA
    if config.load_from is not None:
        if args.load_from is not None:
            config.load_from = args.load_from
        load_pretrained_to_routed(model, config.load_from, logger=logger)

    model_ema = deepcopy(model).eval()
    ema_update(model_ema, model, 0.)

    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).cuda()

    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type)

    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(
            sampler=RandomSampler(dataset),
            dataset=dataset,
            batch_size=config.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=True,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.valid_num
        )
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)
    else:
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.num_workers,
            batch_size=config.train_batch_size,
            shuffle=True
        )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze router + light only
    for module in model.modules():
        if isinstance(module, RoutedFFN):
            for param in module.router.parameters():
                param.requires_grad = True
            for param in module.light.parameters():
                param.requires_grad = True

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameter count:", len(trainable_names))
    for name in trainable_names[:50]:
        print(name)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.optimizer['lr'],
        weight_decay=config.optimizer['weight_decay'],
        eps=config.optimizer['eps']
    )

    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(
            config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
            config.optimizer,
            **config.auto_lr
        )

    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        start_epoch, missing, unexpected = load_checkpoint(
            **config.resume_from,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        logger.warning(f'Resume missing keys: {missing}')
        logger.warning(f'Resume unexpected keys: {unexpected}')

    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    print("MODEL DEVICE:", next(model.parameters()).device)

    train()