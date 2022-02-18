import os
import builtins

import torch.distributed as dist

from search_dg import search_seg_dg_policy
from search_dg_2d import search_seg2d_dg_policy


def search_worker(gpu, ngpus_per_node, config, args):
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

    if config.DATASET.NAME in ['optic']:
        search_seg_dg_policy(gpu, ngpus_per_node, config, args)
    elif config.DATASET.NAME in ['rvs']:
        search_seg2d_dg_policy(gpu, ngpus_per_node, config, args)
