import time
import datetime
from hpman.m import hp
from utils import misc
from engine import train_one_epoch, evaluate
import torch
import numpy as np
import random
import json
from models import build_model
from datasets import build_dataset
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from pathlib import Path


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

    optimizer = build_optimizer(model_without_ddp)
    lr_scheduler = build_scheduler(optimizer)

    dataset_train = build_dataset(data_set='tv_MNIST_train')
    dataset_val = build_dataset(data_set='tv_MNIST_val')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, hp('train.batch_size', 1), drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=hp('train.batch_size'),
                                 sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir) / args.runfile.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = (
        str(output_dir / "latest.pth")
        if (output_dir / "latest.pth").exists()
        else hp('model.fine_tune_from', None)
    )
    args.start_epoch = 1
    if resume:
        print(f"resume from {resume}")
        if resume.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(resume, map_location='cpu')
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print("Evaluate only")
        evaluate(model, criterion, data_loader_val, device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, hp('train.total_epoch', 100)):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'latest.pth']
            if (epoch + 1) % hp('train.checkpoint_interval', 100) == 0:
                checkpoint_paths.append(output_dir / f'epoch-{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                misc.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(model, criterion, data_loader_val, device)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k,  v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and misc.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
