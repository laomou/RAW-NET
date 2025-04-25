import time
import datetime
from hpman.m import hp
from utils import misc
from engine import train_one_epoch
import torch
import numpy as np
import random
import json
from models import build_model
from datasets import build_dataset
from torch.utils.data import DataLoader


def save_checkpoint(args, ckp, name):
    pass


def load_checkpoint(ckp_path):
    pass


def main(args):
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-3)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=0, last_epoch=-1)

    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, hp('train.batch_size'), drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=1)

    print("Start training")
    start_time = time.time()
    for epoch in range(hp('train.start_epoch'), hp('train.total_epoch')):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, epoch)
        lr_scheduler.step()

        if (epoch + 1) % hp('train.checkpoint_interval') == 0:
            pass

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        if misc.is_main_process():
            with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
