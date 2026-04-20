import os
import time

from torch.utils.data import DataLoader

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger


# -----------------------------
# Simple Registry replacement
# -----------------------------
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register_module(self):
        def _register(cls):
            self._module_dict[cls.__name__] = cls
            return cls
        return _register

    def get(self, key):
        if key not in self._module_dict:
            raise KeyError(f"{key} is not registered in {self._name}")
        return self._module_dict[key]


# -----------------------------
# build_from_cfg replacement
# -----------------------------
def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

    cfg = cfg.copy()
    if default_args is not None:
        for k, v in default_args.items():
            cfg.setdefault(k, v)

    if 'type' not in cfg:
        raise KeyError("'type' is required in cfg")

    obj_type = cfg.pop('type')

    if isinstance(obj_type, str):
        cls = registry.get(obj_type)
    else:
        cls = obj_type

    return cls(**cfg)


# -----------------------------
# Dataset registry
# -----------------------------
DATASETS = Registry('datasets')

DATA_ROOT = '/cache/data'


def set_data_root(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)


def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get('type')
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()

    transform = cfg.pop('transform', 'default_train')
    transform = get_transform(transform, resolution)

    dataset = build_from_cfg(
        cfg,
        DATASETS,
        default_args=dict(transform=transform, resolution=resolution, **kwargs)
    )

    logger.info(
        f"Dataset {dataset_type} constructed. "
        f"time: {(time.time() - t):.2f} s, "
        f"length (use/ori): {len(dataset)}/{getattr(dataset, 'ori_imgs_nums', 'N/A')}"
    )

    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if 'batch_sampler' in kwargs:
        return DataLoader(
            dataset,
            batch_sampler=kwargs['batch_sampler'],
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )