import os
import sys
import argparse

from distributed import lanuch_mp_worker
from search import search_worker
from train import train_worker
from test import test_worker

from config.defaults import update_config
from config.defaults import _C as config


def main():
    parser = argparse.ArgumentParser(description='Adversarial AutoAugment')
    # multi-processing
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--gpus', default=1, type=int,
                        help='GPUs to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    parser.add_argument('--smoke_test', action='store_true',
                        help='debug mode')
    parser.add_argument('--mode', default='search',
                        help='[search / train / test]')
    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    # convenient for ablation study
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--seed', default=1023, type=int)

    args = parser.parse_args()
    update_config(config, args)

    if args.mode == 'test':
        test_worker(config)
    elif args.mode == 'search':
        lanuch_mp_worker(search_worker, config, args)
    elif args.mode == 'train':
        lanuch_mp_worker(train_worker, config, args)
    else:
        raise NotImplementedError("Only [search / train / test] are supported.")

    if not args.smoke_test:
        sys.stdout.close()


if __name__ == '__main__':
    main()