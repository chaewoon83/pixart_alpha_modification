import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue, Empty
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.folder import default_loader

from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL
from diffusion.data.datasets.InternalData import InternalData
from diffusion.utils.misc import SimpleTimer
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.data.builder import DATASETS
from diffusion.data import ASPECT_RATIO_512, ASPECT_RATIO_1024


# ---------------------------
# SAFE SAVE FUNCTIONS
# ---------------------------
def save_npz_atomic(save_path_no_ext: str, emb_dict: dict):
    final_path = f"{save_path_no_ext}.npz"
    tmp_path = f"{save_path_no_ext}.tmp.{threading.get_ident()}.npz"

    try:
        np.savez_compressed(tmp_path, **emb_dict)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def is_valid_npz(npz_path: str) -> bool:
    try:
        with np.load(npz_path) as f:
            _ = f["caption_feature"]
            _ = f["attention_mask"]
        return True
    except:
        return False


# ---------------------------
# T5 THREAD WORKER
# ---------------------------
def extract_caption_t5_do(q):
    while True:
        try:
            item = q.get_nowait()
        except Empty:
            break

        try:
            extract_caption_t5_job(item)
        finally:
            q.task_done()


def extract_caption_t5_job(item):
    global mutex
    global t5
    global t5_save_dir
    global progress_bar

    with torch.no_grad():
        caption = item['prompt'].strip()
        if isinstance(caption, str):
            caption = [caption]

        save_path = os.path.join(t5_save_dir, Path(item['path']).stem)
        final_npz = f"{save_path}.npz"

        if os.path.exists(final_npz) and is_valid_npz(final_npz):
            progress_bar.update(1)
            return

        if os.path.exists(final_npz):
            try:
                os.remove(final_npz)
            except:
                pass

        try:
            with mutex:
                caption_emb, emb_mask = t5.get_text_embeddings(caption)

                emb_dict = {
                    'caption_feature': caption_emb.float().cpu().numpy(),
                    'attention_mask': emb_mask.cpu().numpy(),
                }

                save_npz_atomic(save_path, emb_dict)

        except Exception as e:
            print(f"Error processing {item['path']}: {e}")
            if os.path.exists(final_npz):
                try:
                    os.remove(final_npz)
                except:
                    pass

        progress_bar.update(1)


# ---------------------------
# MAIN T5 EXTRACTION
# ---------------------------
def extract_caption_t5():
    global t5
    global t5_save_dir
    global mutex
    global progress_bar

    t5 = T5Embedder(
        device="cuda",
        local_cache=True,
        cache_dir=f'{args.pretrained_models_dir}/t5_ckpts',
        model_max_length=120
    )

    t5_save_dir = args.t5_save_root
    os.makedirs(t5_save_dir, exist_ok=True)

    with open(args.json_path, 'r') as f:
        train_data_json = json.load(f)

    train_data = train_data_json[args.start_index: args.end_index]

    mutex = threading.Lock()
    jobs = Queue()

    for item in train_data:
        jobs.put(item)

    # 🔥 progress bar
    progress_bar = tqdm(
        total=len(train_data),
        desc="Extracting T5 Features",
        dynamic_ncols=True
    )

    workers = []
    num_threads = min(4, max(1, os.cpu_count() or 1))

    for _ in range(num_threads):
        worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,), daemon=True)
        worker.start()
        workers.append(worker)

    jobs.join()
    progress_bar.close()

    for worker in workers:
        worker.join(timeout=1)


# ---------------------------
# VAE EXTRACTION
# ---------------------------
def extract_img_vae():
    vae = AutoencoderKL.from_pretrained(
        f'{args.pretrained_models_dir}/sd-vae-ft-ema'
    ).to(device)

    with open(args.json_path, 'r') as f:
        train_data_json = json.load(f)

    #vae_save_root = f'{args.vae_save_root}/{image_resize}resolution'
    vae_save_root = f'{args.vae_save_root}'
    os.makedirs(vae_save_root, exist_ok=True)

    vae_save_dir = os.path.join(vae_save_root, 'noflip')
    #vae_save_dir =  vae_save_root
    os.makedirs(vae_save_dir, exist_ok=True)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize),
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])

    for item in tqdm(train_data_json, desc="Extracting VAE"):
        image_name = item['path']
        save_path = os.path.join(vae_save_dir, Path(image_name).stem)
        
        final_npy = f"{save_path}.npy"
        tmp_npy = f"{save_path}.tmp.npy"

        if os.path.exists(final_npy):
            continue

        try:
            img = Image.open(f'{args.dataset_root}/{image_name}')
            img = transform(img).to(device)[None]

            with torch.no_grad():
                posterior = vae.encode(img).latent_dist
                z = torch.cat([posterior.mean, posterior.std], dim=1).cpu().numpy()

            np.save(tmp_npy, z)
            os.replace(tmp_npy, final_npy)

        except Exception as e:
            print(e)


# ---------------------------
# ARGS
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--t5_save_root', type=str)
    parser.add_argument('--vae_save_root', type=str)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--pretrained_models_dir', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=1000000, type=int)
    return parser.parse_args()


# ---------------------------
# MAIN
# ---------------------------
if __name__ == '__main__':
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize = args.img_size

    extract_caption_t5()