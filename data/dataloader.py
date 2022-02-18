import torch
import torch.distributed as dist

from data.optic import FundusSegmentation
from data.vessel import RetinalVesselSegmentation
from .transform import get_dg_segtransform, train_collate_fn, \
    train_dg_collate_fn, test_collate_fn, test_dg_collate_fn


def get_seg_dg_dataloader(cfg, args, batch_size, workers):
    transform_train, transform_test = get_dg_segtransform(cfg.DATASET.NAME)
    dataroot = cfg.DATASET.ROOT

    if cfg.DATASET.NAME == 'optic':
        trainset = FundusSegmentation(dataroot, phase='train', splitid=cfg.DATASET.DG.TRAIN, transform=transform_train)
        testset = FundusSegmentation(dataroot, phase='test', splitid=cfg.DATASET.DG.TEST, transform=transform_test)
    if cfg.DATASET.NAME == 'rvs':
        trainset = RetinalVesselSegmentation(dataroot, phase='train', splitid=cfg.DATASET.DG.TRAIN, transform=transform_train)
        testset = RetinalVesselSegmentation(dataroot, phase='test', splitid=cfg.DATASET.DG.TEST, transform=transform_test)

    train_sampler = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, 
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=train_dg_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size if cfg.DATASET.NAME in ['optic', 'rvs'] else 1, shuffle=False, num_workers=workers, pin_memory=True,
        drop_last=False, collate_fn=test_dg_collate_fn if cfg.DATASET.NAME in ['optic', 'rvs'] else train_dg_collate_fn
    )

    return train_sampler, train_loader, test_loader
