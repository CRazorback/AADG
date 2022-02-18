from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


def get_optimizer_scheduler(controller, model, cfg):
    # define controller optimizer
    controller_optimizer = Adam(controller.parameters(), lr=0.00035)
    
    # define optimizer
    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)  
    scheduler = MultiStepLR(optimizer, [cfg.TRAIN.WARMUP_EPOCH], gamma=0.1, last_epoch=-1, verbose=False)
        
    return optimizer, scheduler, controller_optimizer


def get_optimizer_scheduler2(model, cfg):
    # define optimizer
    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
    
    # define scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.END_EPOCH)
        
    return optimizer, scheduler


def get_dis_optimizer_scheduler(discriminator, cfg):
    # define optimizer
    optimizer = Adam(discriminator.parameters(), lr=cfg.TRAIN.LR)
    
    if cfg.TRAIN.WARMUP_EPOCH > 0 and cfg.DISCRIMINATOR.NAME == 'image':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.WARMUP_EPOCH)
    else:
        scheduler = MultiStepLR(optimizer, [cfg.TRAIN.WARMUP_EPOCH], gamma=1, last_epoch=-1, verbose=False)

    return optimizer, scheduler