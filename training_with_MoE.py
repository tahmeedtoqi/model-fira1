# Install necessary modules
!pip install safetensors

import os
import time
import math
import pickle
from contextlib import nullcontext
import csv
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import json
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import safetensors.torch as st

ded

# -----------------------------------------------------------------------------
# Default config values
out_dir = '/kaggle/working/'
eval_interval = 100
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'resume'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'FIRA'
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 512
n_layer = 8
n_head = 8
n_embd = 512
d_ff = 4 * n_embd
dropout = 0.0
num_experts = 4
learning_rate = 6e-4
max_iters = 1000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 2000
min_lr = 5e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if 'num_experts' not in config_keys:
    config_keys.append('num_experts')
if 'd_ff' not in config_keys:
    config_keys.append('d_ff')
config = {k: globals()[k] for k in config_keys}

# Load custom tokenizer from JSON
tokenizer_path = '/kaggle/input/tokenz1/tokenizer.json'
with open(tokenizer_path, 'r') as f:
    tokenizer_data = json.load(f)
    try:
        my_vocab = tokenizer_data['model']['vocab']
        my_vocab_size = len(my_vocab)
    except KeyError:
        print("Available keys:", tokenizer_data.keys())
        raise KeyError("Adjust the key based on the printed structure")
print(f"Loaded custom tokenizer with vocab_size = {my_vocab_size}")

token_to_id = {token: idx for idx, token in enumerate(my_vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

def encode(text):
    return [token_to_id.get(token, 0) for token in text.split()]

def decode(ids):
    return ' '.join([id_to_token.get(id, '<unk>') for id in ids])

# -----------------------------------------------------------------------------
# Initialization and setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader
data_dir = os.path.join('/kaggle/input/', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    x = torch.clamp(x, 0, my_vocab_size - 1)
    y = torch.clamp(y, 0, my_vocab_size - 1)
    return x, y

causal_mask = torch.tril(torch.ones((block_size, block_size), device=device)).view(1, 1, block_size, block_size)

# Model initialization
model_args = dict(
    num_layers=n_layer,
    n_head=n_head,
    d_model=n_embd,
    d_ff=d_ff,
    num_experts=num_experts,
    max_seq_len=block_size,
    dropout=dropout,
    vocab_size=my_vocab_size
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model = FIRA(**model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    if checkpoint_model_args['vocab_size'] != my_vocab_size:
        raise ValueError(f"Checkpoint vocab_size mismatch")
    model = FIRA(**checkpoint_model_args)
    state_dict = st.load_file(os.path.join(out_dir, 'model.safetensors'), device=device)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout, num_experts=num_experts)
    model = FIRA.from_pretrained(init_from, num_experts=num_experts, override_args=override_args)
    model_args = model.config.copy()

if init_from != 'resume':
    iter_num = 0
    best_val_loss = float('inf')

model.to(device)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile model if enabled
if compile:
    print("Compiling the model... (takes a ~minute)")
    try:
        unoptimized_model = model
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled successfully")
    except Exception as e:
        print(f"Compilation failed: {e}")
        print("Falling back to eager mode")
        compile = False

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X, mask=causal_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

log_file = os.path.join(out_dir, 'training_log.csv')

def log_to_csv(iter_num, train_loss=None, val_loss=None, lr=None, mfu=None):
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['iter', 'train_loss', 'val_loss', 'lr', 'mfu']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'iter': iter_num}
        if train_loss is not None:
            row['train_loss'] = train_loss
        if val_loss is not None:
            row['val_loss'] = val_loss
        if lr is not None:
            row['lr'] = lr
        if mfu is not None:
            row['mfu'] = mfu
        writer.writerow(row)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        log_to_csv(iter_num, train_loss=losses['train'], val_loss=losses['val'], lr=lr)
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                state_dict = raw_model.state_dict()
                st.save_file(state_dict, os.path.join(out_dir, 'model.safetensors'))
                checkpoint = {
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))
        
        prompt = "Hello, I am a language model"
        input_ids = torch.tensor([encode(prompt)], device=device)
        generated_ids = raw_model.generate(input_ids, max_new_tokens=20, temperature=0.7, top_k=50)
        generated_text = decode(generated_ids[0].tolist())
        print(f"Generated text: {generated_text}")

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits = model(X, mask=causal_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        log_to_csv(iter_num, train_loss=lossf, lr=lr, mfu=running_mfu*100)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# Plotting with type conversion
if master_process:
    df = pd.read_csv(log_file)
    # Convert loss columns to float, handling potential non-numeric values
    for col in ['train_loss', 'val_loss']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, NaN for invalid entries
    
    plt.figure(figsize=(10, 5))
    if 'train_loss' in df.columns and not df['train_loss'].isna().all():
        plt.plot(df['iter'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns and not df['val_loss'].isna().all():
        plt.plot(df['iter'], df['val_loss'], label='Val Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig(os.path.join(out_dir, 'training_progress.png'))
    plt.close()
    print(f"Training progress plot saved to {os.path.join(out_dir, 'training_progress.png')}")

if ddp:
    destroy_process_group()