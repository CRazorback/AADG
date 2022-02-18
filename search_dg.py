import os
import time
import torch
import json
import utils

import numpy as np

import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1
from data.dataloader import get_seg_dg_dataloader
from data.policy import DGMultiPolicy, RandAugment, parse_policies 
from models import load_ddp_controller, load_ddp_discriminator, load_ddp_model
from scheduler import get_dis_optimizer_scheduler, get_optimizer_scheduler
from losses import search_loss, task_loss, CrossEntropy

from geomloss import SamplesLoss
from medpy.metric import binary


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
    length = len(train_loader)

    end = time.time()

    for i, sample in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # compute the output    
        input = sample['image'].cuda(non_blocking=True)
        mask_gt = sample['label'].cuda(non_blocking=True)
        domain_gt = sample['dc'].cuda(non_blocking=True)

        seg_output, feature = model(input)
        dis_output = discriminator(feature.detach())

        seg_soft = torch.sigmoid(seg_output)
        seg_loss = model_criterion(seg_soft, mask_gt)
        dis_loss = dis_criterion(dis_output, domain_gt)
        cup_dsc = f1_score(torch.stack([1 - seg_soft[:,0], seg_soft[:,0]], dim=1), mask_gt[:,0].long())[1]
        disc_dsc = f1_score(torch.stack([1 - seg_soft[:,1], seg_soft[:,1]], dim=1), mask_gt[:,1].long())[1]

        # optimize segmentation model
        model_optimizer.zero_grad()
        seg_loss.backward()
        model_optimizer.step()
        # optimize discriminator
        dis_optimizer.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

        seg_losses.update(seg_loss.item(), input.size(0))
        dis_losses.update(dis_loss.item(), input.size(0))
        train_cup_dsc.update(cup_dsc.item(), input.size(0))
        train_disc_dsc.update(disc_dsc.item(), input.size(0))
        batch_time.update(time.time() - end)

        if i % config.PRINT_FREQ == 0 and logger:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Seg Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t' \
                  'Dis Loss {dis_loss.val:.5f} ({dis_loss.avg:.5f})\t'.format(
                      epoch, i, length, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, seg_loss=seg_losses,
                      dis_loss=dis_losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_seg_loss', seg_losses.val, global_steps)
                writer.add_scalar('train_dis_loss', dis_losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    
    if logger:
        msg = 'Train Epoch {} time:{:.4f} seg loss:{:.4f} dis loss:{:.4f} dsc@cup:{:.4f} dsc@disc:{:.4f}'\
            .format(epoch, batch_time.avg, seg_losses.avg, dis_losses.avg, train_cup_dsc.avg, train_disc_dsc.avg)
        logger.info(msg)


def train(config, train_loader, model, discriminator, model_criterion,
          dis_criterion, model_optimizer, dis_optimizer, M, epoch, writer_dict, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    seg_losses = utils.AverageMeter()
    dis_losses = utils.AverageMeter()
    div_ot_monitor = utils.AverageMeter()
    train_cup_dsc = utils.AverageMeter()
    train_disc_dsc = utils.AverageMeter()

    f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')

    model.train()
    discriminator.train()
    sinkhorn = SamplesLoss("sinkhorn", cost='( IntCst(1) - (X | Y) / ( Norm2(X) * Norm2(Y) ) )', backend='online')
    rewards = torch.zeros(M).cuda()
    rewards.requires_grad = False
    length = len(train_loader)

    end = time.time()

    for i, sample in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # compute the output    
        input = sample['aug_images'].cuda(non_blocking=True)
        mask_gt = sample['aug_labels'].cuda(non_blocking=True)
        domain_gt = sample['dc'].cuda(non_blocking=True)

        seg_output, feature = model(input)
        # action
        dis_output, domain_feature = discriminator(feature.detach(), momentum=True, return_feature=True)
        domain_feature = domain_feature.detach()
        # bp
        dis_output_bp = discriminator(feature.detach(), momentum=False)
        dis_loss_bp = dis_criterion(dis_output_bp, domain_gt)

        seg_soft = torch.sigmoid(seg_output)
        seg_loss_list = [model_criterion(seg_soft[j::M], mask_gt[j::M]) for j in range(M)]
        seg_loss = torch.mean(torch.stack(seg_loss_list))
        dis_loss_list = [dis_criterion(dis_output[j::M], domain_gt[j::M]) for j in range(M)]
        dis_loss = torch.mean(torch.stack(dis_loss_list))

        cup_dsc = 0
        disc_dsc = 0
        diversity_ot = 0
        # compute reward and accuracy for ddp
        for j, _loss in enumerate(dis_loss_list):
            domain_feature_sub = domain_feature[j::M]
            domain_gt_sub = domain_gt[j::M]
            domain_idx = torch.argmax(domain_gt_sub, dim=1)
            domain1_idx = (domain_idx == 0).nonzero(as_tuple=True)
            domain2_idx = (domain_idx == 1).nonzero(as_tuple=True)
            domain3_idx = (domain_idx == 2).nonzero(as_tuple=True)
            domain1_fe, domain2_fe, domain3_fe = domain_feature_sub[domain1_idx], domain_feature_sub[domain2_idx], domain_feature_sub[domain3_idx]
            dist_12 = sinkhorn(domain1_fe, domain2_fe)
            dist_23 = sinkhorn(domain2_fe, domain3_fe)
            dist_13 = sinkhorn(domain1_fe, domain3_fe)
            rewards[j] += (dist_12 + dist_13 + dist_23)
            diversity_ot += (dist_12 + dist_13 + dist_23)

            _cup_dsc = f1_score(torch.stack([1 - seg_soft[:,0], seg_soft[:,0]], dim=1), mask_gt[:,0].long())[1]
            _disc_dsc = f1_score(torch.stack([1 - seg_soft[:,1], seg_soft[:,1]], dim=1), mask_gt[:,1].long())[1]
            cup_dsc += _cup_dsc / M
            disc_dsc += _disc_dsc / M

        # optimize segmentation model
        model_optimizer.zero_grad()
        seg_loss.backward()
        model_optimizer.step()
        # optimize discriminator
        dis_optimizer.zero_grad()
        dis_loss_bp.backward()
        dis_optimizer.step()

        seg_losses.update(seg_loss.item(), input.size(0))
        dis_losses.update(dis_loss.item(), input.size(0))
        div_ot_monitor.update(diversity_ot.item(), input.size(0))
        train_cup_dsc.update(cup_dsc.item(), input.size(0))
        train_disc_dsc.update(disc_dsc.item(), input.size(0))
        batch_time.update(time.time() - end)

        if i % config.PRINT_FREQ == 0 and logger:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Seg Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t' \
                  'Dis Loss {dis_loss.val:.5f} ({dis_loss.avg:.5f})\t'.format(
                      epoch, i, length, batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, seg_loss=seg_losses,
                      dis_loss=dis_losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_seg_loss', seg_losses.val, global_steps)
                writer.add_scalar('train_dis_loss', dis_losses.val, global_steps)
                writer.add_scalar('diversity_ot_distance', diversity_ot.item(), global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    
    if logger:
        msg = 'Train Epoch {} time:{:.4f} seg loss:{:.4f} dis loss:{:.4f} dsc@cup:{:.4f} dsc@disc:{:.4f} OT:{:.4f}'\
            .format(epoch, batch_time.avg, seg_losses.avg, dis_losses.avg, train_cup_dsc.avg, train_disc_dsc.avg, div_ot_monitor.avg)
        logger.info(msg)

    # normalize award
    return (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-5)


def validate(config, val_loader, model, epoch, writer_dict, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    val_cup_dsc = utils.AverageMeter()
    val_cup_hd = utils.AverageMeter()
    val_disc_dsc = utils.AverageMeter()
    val_disc_hd = utils.AverageMeter()

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

            seg_output, _ = model(input)

            seg_soft = torch.sigmoid(seg_output)
            seg_hard = torch.tensor(seg_soft.clone().detach() > 0.75).float()
            # dice similarity coefficient
            cup_dsc = f1_score(torch.stack([1 - seg_hard[:,0], seg_hard[:,0]], dim=1), mask_gt[:,0].long())[1]
            disc_dsc = f1_score(torch.stack([1 - seg_hard[:,1], seg_hard[:,1]], dim=1), mask_gt[:,1].long())[1]
            # hasudorff distance
            total_cup_hd = 0
            total_disc_hd = 0
            for idx in range(input.size(0)):
                binary_output = (seg_hard.cpu().numpy()[idx, 0]).astype(np.bool)
                target_np = (mask_gt.cpu().numpy()[idx, 0]).astype(np.bool)
                if binary_output.astype(np.uint8).sum() < 1e-4:
                    total_cup_hd += 100
                else:
                    total_cup_hd += binary.hd95(binary_output, target_np)
                binary_output = (seg_hard.cpu().numpy()[idx, 1]).astype(np.bool)
                target_np = (mask_gt.cpu().numpy()[idx, 1]).astype(np.bool)
                if binary_output.astype(np.uint8).sum() < 1e-4:
                    total_disc_hd += 100
                else:
                    total_disc_hd += binary.hd95(binary_output, target_np)

            val_cup_dsc.update(cup_dsc.item(), input.size(0))
            val_disc_dsc.update(disc_dsc.item(), input.size(0))
            val_cup_hd.update(total_cup_hd / input.size(0), input.size(0))
            val_disc_hd.update(total_disc_hd / input.size(0), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if writer_dict and logger:
        msg = 'Test Epoch {} time:{:.4f} dsc@cup:{:.4f} dsc@disc:{:.4f} HD@cup:{:.4f} HD@disc:{:.4f}'\
            .format(epoch, batch_time.avg, val_cup_dsc.avg, val_disc_dsc.avg, val_cup_hd.avg, val_disc_hd.avg)
        logger.info(msg)

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_cup_dsc', val_cup_dsc.avg, global_steps)
        writer.add_scalar('valid_disc_dsc', val_disc_dsc.avg, global_steps)
        writer.add_scalar('valid_cup_hd', val_cup_hd.avg, global_steps)
        writer.add_scalar('valid_disc_hd', val_disc_hd.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return val_cup_dsc.avg, val_disc_dsc.avg, val_cup_hd.avg, val_disc_hd.avg


def search_seg_dg_policy(gpu, ngpus_per_node, config, args):
    model, batch_size, workers = load_ddp_model(ngpus_per_node, args, config)
    controller, M, _ = load_ddp_controller(ngpus_per_node, args, config)
    discriminator, _, _ = load_ddp_discriminator(ngpus_per_node, args, config)
    train_samplers, train_loader, test_loader = get_seg_dg_dataloader(config, args, batch_size, workers)
    model_optimizer, model_lrscheduler, controller_optimizer = get_optimizer_scheduler(controller, model, config)
    dis_optimizer, dis_lrscheduler = get_dis_optimizer_scheduler(discriminator, config)
    model_criterion = task_loss(config)
    controller_criterion = search_loss(config)
    dis_criterion = CrossEntropy()

    controller_criterion.register_optimizer(controller_optimizer)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    best_dsc = 0
    best_metric = {'epoch': 0, 'avg_dsc': 0, 'cup_dsc': 0, 'disc_dsc': 0, 'avg_hd': 0, 'cup_hd': 0, 'disc_hd': 0}
    mag_probs_trajectory = []
    op_probs_trajectory = []

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

        if config.TRAIN.WARMUP_EPOCH > epoch:
            train_loader.dataset.transforms.transforms[0] = RandAugment(N=2)
            pretrain(config, train_loader, model, discriminator, model_criterion,
                     dis_criterion, model_optimizer, dis_optimizer, epoch, writer_dict, logger)
            model_lrscheduler.step()
            dis_lrscheduler.step()
        else:
            # share weights
            if config.TRAIN.WARMUP_EPOCH == epoch:
                discriminator.synchronize_parameters()
            # sample augmentation policies
            controller.train()
            policies, op_probs, mag_probs, log_probs, entropies = controller(M)
            parsed_policies = parse_policies(policies.cpu().detach().numpy(), config, logger)
            train_loader.dataset.transforms.transforms[0] = DGMultiPolicy(parsed_policies)

            # train
            normalized_rewards = train(config, train_loader, model, discriminator, model_criterion,
                                       dis_criterion, model_optimizer, dis_optimizer, M, epoch, writer_dict, logger)
            discriminator.momentum_update()
            controller_loss, score_loss, entropy_penalty = controller_criterion(controller, policies, log_probs, entropies, normalized_rewards)

            model_lrscheduler.step()
            dis_lrscheduler.step()

        # evaluate
        cup_dsc, disc_dsc, cup_hd, disc_hd = validate(config, test_loader, model, epoch, writer_dict, logger)
        dsc = (cup_dsc + disc_dsc) / 2
        hd = (cup_hd + disc_hd) / 2
        is_best = dsc > best_dsc
        if is_best:
            best_dsc = max(dsc, best_dsc)
            best_metric = {'epoch': epoch + 1, 'avg_dsc': dsc, 'cup_dsc': cup_dsc, 'disc_dsc': disc_dsc, 'avg_hd': hd, 'cup_hd': cup_hd, 'disc_hd': disc_hd}

        if mp_flag: 
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            logger.info('=> best: {}'.format(str(is_best)))

            if config.TRAIN.WARMUP_EPOCH <= epoch:
                logger.info(mag_probs.detach().cpu().numpy())
                logger.info(op_probs.detach().cpu().numpy())
                mag_probs_trajectory.append(mag_probs.detach().cpu().numpy())
                op_probs_trajectory.append(op_probs.detach().cpu().numpy())

                logger.info('Train Epoch {}: controller loss:{:.4f} score loss:{:.4f} entropy penalty:{:.4f}'.format(
                    epoch, controller_loss.detach().cpu().numpy(), score_loss.detach().cpu().numpy(), entropy_penalty.detach().cpu().numpy()))
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar('controller_loss', controller_loss.item(), global_steps)
                writer.add_scalar('score_loss', score_loss.item(), global_steps)
                writer.add_scalar('entropy_penalty', entropy_penalty.item(), global_steps)

                utils.save_checkpoint(
                    {
                        "state_dict": model,
                        "epoch": epoch + 1,
                        "best_dsc": best_dsc,
                        "optimizer": model_optimizer.state_dict(),
                        "policies": parsed_policies
                    }, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    if mp_flag:
        final_model_state_file = os.path.join(final_output_dir,
                                              'final_model_state.pth')
        final_controller_state_file = os.path.join(final_output_dir,
                                              'final_controller_state.pth')
        logger.info('saving final model and controller state to {} and {}'.format(
            final_model_state_file, final_controller_state_file))
        torch.save(model.state_dict(), final_model_state_file)
        torch.save(controller.state_dict(), final_controller_state_file)
        writer_dict['writer'].close()
        # save trajactory
        np.save(os.path.join(final_output_dir, 'mag_probs_trajectory.npy'), np.array(mag_probs_trajectory))
        np.save(os.path.join(final_output_dir, 'op_probs_trajectory.npy'), np.array(op_probs_trajectory))
        # final result
        logger.info('Best Epoch: {}, dsc@cup:{:.4f} dsc@disc:{:.4f} HD@cup:{:.4f} HD@disc:{:.4f}'.format(
            best_metric['epoch'], best_metric['cup_dsc'], best_metric['disc_dsc'], best_metric['cup_hd'], best_metric['disc_hd']))
        # save final result
        results = json.dumps(best_metric)
        with open(os.path.join(final_output_dir, 'final_result.json'), 'w') as f:
            f.write(results)
