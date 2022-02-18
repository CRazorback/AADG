import os
import builtins
import torch

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from train_dg import train_dg_seg_network
from train_dg_2d import train_dg_2d_seg_network


def train_worker(gpu, ngpus_per_node, config, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    elif config.DATASET.NAME in ['optic']:
        train_dg_seg_network(gpu, ngpus_per_node, config, args)
    elif config.DATASET.NAME in ['rvs']:
        train_dg_2d_seg_network(gpu, ngpus_per_node, config, args)