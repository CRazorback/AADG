import os
import time
import torch
import json
import utils

import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1
from data.dataloader import get_seg_dg_dataloader
from models import load_ddp_discriminator, load_ddp_model
from scheduler import get_dis_optimizer_scheduler, get_optimizer_scheduler2
from losses import task_loss, DGLSGAN


def pretrain(config, train_loader, model, discriminator, model_criterion,
             dis_criterion, model_optimizer, dis_optimizer, epoch, writer_dict, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    seg_losses = utils.AverageMeter()
    dis_losses = utils.AverageMeter()
    train_cup_dsc = utils.AverageMeter()
    train_disc_dsc = utils.AverageMeter()

    f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')

    model.train()
    discriminator.train()

    end = time.time()

    for i, sample in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # compute the output    
        input = sample['image'].cuda(non_blocking=True)
        mask_gt = sample['label'].cuda(non_blocking=True)

        seg_output = model(input)

        seg_soft = torch.sigmoid(seg_output)
        seg_loss = model_criterion(seg_soft, mask_gt)
        cup_dsc = f1_score(torch.stack([1 - seg_soft[:,0], seg_soft[:,0]], dim=1), mask_gt[:,0].long())[1]
        disc_dsc = f1_score(torch.stack([1 - seg_soft[:,1], seg_soft[:,1]], dim=1), mask_gt[:,1].long())[1]

        # optimize segmentation model
        model_optimizer.zero_grad()
        seg_loss.backward()
        model_optimizer.step()

        seg_losses.update(seg_loss.item(), input.size(0))
        train_cup_dsc.update(cup_dsc.item(), input.size(0))
        train_disc_dsc.update(disc_dsc.item(), input.size(0))
        batch_time.update(time.time() - end)

        if i % config.PRINT_FREQ == 0 and logger:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Seg Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, seg_loss=seg_losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_seg_loss', seg_losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    
    if logger:
        msg = 'Train Epoch {} time:{:.4f} seg loss:{:.4f} dis loss:{:.4f} dsc@cup:{:.4f} dsc@disc:{:.4f}'\
            .format(epoch, batch_time.avg, seg_losses.avg, dis_losses.avg, train_cup_dsc.avg, train_disc_dsc.avg)
        logger.info(msg)


def validate(config, val_loader, model, epoch, writer_dict, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    val_cup_dsc = utils.AverageMeter()
    val_disc_dsc = utils.AverageMeter()

    f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            # measure data time
            data_time.update(time.time() - end)

            # compute the output    
            input = sample['image'].cuda(non_blocking=True)
            mask_gt = sample['label'].cuda(non_blocking=True)
            # domain_gt = sample['dc'].cuda(non_blocking=True)

            seg_output = model(input)

            seg_soft = torch.sigmoid(seg_output)
            seg_hard = torch.tensor(seg_soft.clone().detach() > 0.75).float()
            # seg_hard = seg_soft.clone().detach()
            cup_dsc = f1_score(torch.stack([1 - seg_hard[:,0], seg_hard[:,0]], dim=1), mask_gt[:,0].long())[1]
            disc_dsc = f1_score(torch.stack([1 - seg_hard[:,1], seg_hard[:,1]], dim=1), mask_gt[:,1].long())[1]

            val_cup_dsc.update(cup_dsc.item(), input.size(0))
            val_disc_dsc.update(disc_dsc.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if writer_dict and logger:
        msg = 'Test Epoch {} time:{:.4f} dsc@cup:{:.4f} dsc@disc:{:.4f}'\
            .format(epoch, batch_time.avg, val_cup_dsc.avg, val_disc_dsc.avg)
        logger.info(msg)

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_cup_dsc', val_cup_dsc.avg, global_steps)
        writer.add_scalar('valid_disc_dsc', val_disc_dsc.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return val_cup_dsc.avg, val_disc_dsc.avg


def train_dg_seg_network(gpu, ngpus_per_node, config, args):
    model, batch_size, workers = load_ddp_model(ngpus_per_node, args, config)
    discriminator, _, _ = load_ddp_discriminator(ngpus_per_node, args, config)
    train_samplers, train_loader, test_loader = get_seg_dg_dataloader(config, args, batch_size, workers)
    model_optimizer, model_lrscheduler = get_optimizer_scheduler2(model, config)
    dis_optimizer, dis_lrscheduler = get_dis_optimizer_scheduler(discriminator, config)
    model_criterion = task_loss(config)
    dis_criterion = DGLSGAN(config)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    best_dsc = 0
    best_metric = {'epoch': 0, 'avg_dsc': 0, 'cup_dsc': 0, 'disc_dsc': 0}

    # only enable tensorboard for main process
    mp_flag = not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0)
    if mp_flag:
        logger, final_output_dir, tb_log_dir = \
            utils.create_logger(config, args.cfg, 'train')
        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None
        logger = None

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        if args.distributed:
            for train_sampler in train_samplers:
                train_sampler.set_epoch(epoch)

        pretrain(config, train_loader, model, discriminator, model_criterion,
                 dis_criterion, model_optimizer, dis_optimizer, epoch, writer_dict, logger)
        model_lrscheduler.step()

        # evaluate
        cup_dsc, disc_dsc = validate(config, test_loader, model, epoch, writer_dict, logger)
        dsc = (cup_dsc + disc_dsc) / 2

        is_best = dsc > best_dsc
        if is_best:
            best_dsc = max(dsc, best_dsc)
            best_metric = {'epoch': epoch + 1, 'avg_dsc': dsc, 'cup_dsc': cup_dsc, 'disc_dsc': disc_dsc}

        if mp_flag:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            logger.info('=> best: {}'.format(str(is_best)))
            utils.save_checkpoint(
                {
                    "state_dict": model,
                    "epoch": epoch + 1,
                    "best_dsc": best_dsc,
                    "optimizer": model_optimizer.state_dict()
                }, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    if mp_flag:
        final_model_state_file = os.path.join(final_output_dir,
                                              'final_state.pth')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.state_dict(), final_model_state_file)
        writer_dict['writer'].close()
        # final result
        logger.info('Best Epoch: {}, dsc@cup:{:.4f} dsc@disc:{:.4f}'.format(
            best_metric['epoch'], best_metric['cup_dsc'], best_metric['disc_dsc']))
        # save final result
        results = json.dumps(best_metric)
        with open(os.path.join(final_output_dir, 'final_result.json'), 'w') as f:
            f.write(results)