_base_ = ['../PixArt_xl2_internal.py']
data_root = 'data'
image_list_json = ['data_info.json',]
dataset_alias = 'coco'
validation_prompts = None

data = dict(
    type='InternalData',
    root='coco_train',
    image_list_json=image_list_json,
    transform='default_train',
    load_vae_feat=False
)

image_size = 256

# model setting
window_block_indexes=[]
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2_Selective'
fp32_attention = False
load_from = None
vae_pretrained = "stabilityai/sd-vae-ft-ema"
# training setting
eval_sampling_steps = 200

num_workers= 2
train_batch_size = 4 # 32  # max 96 for PixArt-L/4 when grad_checkpoint
num_epochs = 3 # 3
gradient_accumulation_steps = 1
grad_checkpointing = False
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=100)

log_interval = 10
save_model_epochs=1
