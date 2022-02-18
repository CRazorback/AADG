import torch
import segmentation_models_pytorch as smp

from .controller import Controller
from .discriminator import FeatureDiscriminator, MomentumFeatureDiscriminator, ImageDiscriminator


def load_ddp_model(ngpus_per_node, args, cfg):
    name = cfg.MODEL.NAME
    backbone = cfg.MODEL.BACKBONE
    num_classes = class_parser(cfg.DATASET.NAME)

    print("=> creating model '{}' with '{}".format(name, backbone))
    
    if name == 'deeplabv3+':
        assert backbone in ['mobilenet_v2']
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=num_classes,
            aux_params=dict(pooling='avg') if 'feature' in cfg.DISCRIMINATOR.NAME else None                       
        )
    else:
        raise NotImplementedError(name + ' has not been implemented!')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = int(cfg.TRAIN.BATCH_SIZE / ngpus_per_node)
            workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        batch_size = cfg.TRAIN.BATCH_SIZE
        workers = args.workers
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    return model, batch_size, workers


def load_model(args, cfg):
    name = cfg.MODEL.NAME
    backbone = cfg.MODEL.BACKBONE
    num_classes = class_parser(cfg.DATASET.NAME)

    print("=> creating model '{}' with '{}".format(name, backbone))
    
    if name == 'deeplabv3+':
        assert backbone in ['mobilenet_v2']
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",     
            in_channels=3,                  
            classes=num_classes,
            aux_params=dict(pooling='avg') if 'feature' in cfg.DISCRIMINATOR.NAME else None                       
        )
    else:
        raise NotImplementedError(name + ' has not been implemented!')

    batch_size = cfg.TEST.BATCH_SIZE
    workers = args.workers // args.gpus

    return model, batch_size, workers


def load_ddp_controller(ngpus_per_node, args, cfg):
    name = cfg.CONTROLLER.NAME
    print("=> creating RNN {}".format(name))
    
    if name == 'controller':
        model = Controller(cfg)
    else:
        raise NotImplementedError(name + 'has not been implemented!')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = cfg.CONTROLLER.M
            workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        batch_size = cfg.CONTROLLER.M
        workers = args.workers
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    return model, batch_size, workers


def load_controller(args, cfg):
    name = cfg.CONTROLLER.NAME
    print("=> creating RNN {}".format(name))
    
    if name == 'controller':
        model = Controller(cfg)
    else:
        raise NotImplementedError(name + 'has not been implemented!')

    batch_size = cfg.CONTROLLER.M
    workers = args.workers // args.gpus

    return model, batch_size, workers


def load_ddp_discriminator(ngpus_per_node, args, cfg):
    name = cfg.DISCRIMINATOR.NAME
    num_classes = domain_parser(cfg.DATASET.NAME)
    in_channels = channel_parser(cfg.MODEL.BACKBONE)
    print("=> creating discriminator '{}'".format(name))
    
    if name == 'feature':
        model = FeatureDiscriminator(num_classes)
    elif name == 'image':
        model = ImageDiscriminator(num_classes)
    elif name == 'momentum_feature':
        model = MomentumFeatureDiscriminator(num_classes, in_channels)
    else:
        raise NotImplementedError(name + 'has not been implemented!')

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = int(cfg.TRAIN.BATCH_SIZE / ngpus_per_node)
            workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        batch_size = cfg.CONTROLLER.M
        workers = args.workers
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    return model, batch_size, workers


def load_discriminator(args, cfg):
    name = cfg.DISCRIMINATOR.NAME
    num_classes = domain_parser(cfg.DATASET.NAME)
    in_channels = channel_parser(cfg.MODEL.BACKBONE)
    print("=> creating discriminator '{}'".format(name))
    
    if name == 'feature':
        model = FeatureDiscriminator(num_classes)
    elif name == 'image':
        model = ImageDiscriminator(num_classes)
    elif name == 'momentum_feature':
        model = MomentumFeatureDiscriminator(num_classes, in_channels)
    else:
        raise NotImplementedError(name + 'has not been implemented!')

    batch_size = cfg.TRAIN.BATCH_SIZE
    workers = args.workers // args.gpus

    return model, batch_size, workers


def class_parser(dataset):
    return {
        'rvs': 1,
        'optic': 2,
    }[dataset]


def domain_parser(dataset):
    return {
        'optic': 3,
        'rvs': 3,
    }[dataset]


def channel_parser(backbone):
    return {
        'mobilenet_v2': 1280,
    }[backbone]