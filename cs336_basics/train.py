import os
import json
import torch
import pathlib

from joblib.testing import param
from tqdm import tqdm
from .models import BasicTransformerLM
from utils import *
import numpy as np
import argparse

DATA_DIR = r""
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.dat")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.dat")
CONFIG_PATH = r"./config.json"

def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode='r')
    return arr

def memmap_val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N - context_length - 1) // batch_size
    for bi in range(nb):
        base = bi * batch_size
        x = np.stack([memmap_arr[i: i+context_length] for i in range(base, base + batch_size)])
        y = np.stack([memmap_arr[i+1, i+context_length+1] for i in range(base, base + batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = BasicTransformerLM(**config["model"])

    params = {}
    for group in config.values():
        params.update(group)

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = DotDict(params)

    model, device = to_device_and_compile(model)

    os.makedirs(args.save_path, exist_ok=True)

    train_data = get_memmap_dataset(TRAIN_DATA_PATH)
    val_data = get_memmap_dataset(VAL_DATA_PATH)

    optimizer = AdamWOptimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume_checkpoint:
        print("=> loading checkpoint '{}'".format(args.resume_checkpoint))
        resume_checkpoint_path = os.path.join(args.save_path, f"checkpoint_{args.resume_checkpoint}iter.pth")
        start_iter = load_checkpoint(resume_checkpoint_path, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    for iteration in tqdm(range(start_iter, args.max_iter)):
        model.train()
        # x size [bs, context_length] y size [bs, context_length, vocab_size]
        x, y = get_batch(train_data, args.batch_size, args.context_length, device=device)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.clip_grad_norm)

        lr = cosine_learning_rate_schedule_with_warmup(
            iteration,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_iters
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if (iteration + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in val_data:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    val_loss = cross_entropy_loss(val_logits.reshape(-1, val_logits.shape[-1]), y_val.reshape(-1))
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.val_batches:
                        break

                val_loss_mean = np.mean(val_losses)
                print(f"iteration {iteration}, val loss: {val_loss_mean}")
        if (iteration + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_path, f"checkpoint_{iteration}iter.pth")
            save_checkpoint(model, optimizer, iteration+1, ckpt_path)
            print(f"checkpoint saved to {ckpt_path}")

if __name__ == '__main__':
    main()


