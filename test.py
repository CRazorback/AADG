from collections import OrderedDict
import os
import time
import torch
import json
import utils
import imageio

import numpy as np
import pandas as pd

import torch.utils.data
import torch.utils.data.distributed

from pathlib import Path
from torchmetrics import F1

from data.dataloader import get_seg_dg_dataloader
from data.policy import DGMultiPolicy, parse_policies 
from models import load_controller, load_model


def inference(train_loader, model, output_dir, augment=False):
    model.eval()
    f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')
    output_dict = {'name': [], 'f1_score': []}

    with torch.no_grad():
        for i, sample in enumerate(train_loader):
            # compute the output    
            if augment:
                input = sample['aug_images'].cuda(non_blocking=True)
            else:
                input = sample['image'].cuda(non_blocking=True)
            mask_gt = sample['label'].cuda(non_blocking=True)
        
            seg_output = model(input)
            if isinstance(seg_output, tuple):
                seg_output = seg_output[0]
            seg_soft = torch.sigmoid(seg_output)
            seg_hard = (seg_soft > 0.5).float().clone().detach()

            for j, name in enumerate(sample['img_name']):
                pred = torch.stack([1 - seg_hard[j,0], seg_hard[j,0]], dim=0).unsqueeze(0)
                gt = mask_gt[j,0].unsqueeze(0).long()
                dsc = f1_score(pred, gt)[1]
                imageio.imsave(os.path.join(output_dir, name[:-3]+'jpg'), (seg_hard[j].cpu().numpy()[0]*255).astype(np.uint8))
                output_dict['name'].append(name)
                output_dict['f1_score'].append(dsc.item())
                print('Saving {}...'.format(name))

    pd.DataFrame.from_dict(output_dict).to_csv(os.path.join(output_dir, 'test_result.csv'), index=False)


def optic_inference(test_loader, model, output_dir, augment=False):
    model.cuda()
    model.eval()
    f1_score = F1(num_classes=2, average=None, mdmc_average='samplewise')
    output_dict = {'name': [], 'f1_score_avg': [], 'f1_score_disc': [], 'f1_score_cup': []}

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            # compute the output    
            input = sample['image'].cuda(non_blocking=True)
            mask_gt = sample['label'].cuda(non_blocking=True)

            seg_output = model(input)
            if isinstance(seg_output, tuple):
                seg_output = seg_output[0]
            seg_soft = torch.sigmoid(seg_output)
            seg_hard = torch.tensor(seg_soft.clone().detach() > 0.75).float()

            for j, name in enumerate(sample['img_name']):
                # dice similarity coefficient
                pred_cup = torch.stack([1 - seg_hard[j,0], seg_hard[j,0]], dim=0).unsqueeze(0)
                pred_disc = torch.stack([1 - seg_hard[j,1], seg_hard[j,1]], dim=0).unsqueeze(0)
                gt_cup = mask_gt[j,0].unsqueeze(0).long()
                gt_disc = mask_gt[j,1].unsqueeze(0).long()
                dsc_cup = f1_score(pred_cup, gt_cup)[1]
                dsc_disc = f1_score(pred_disc, gt_disc)[1]
                seg_map = torch.zeros_like(mask_gt[j,0])
                disc_map = torch.where(seg_hard[j,1] == 1, torch.ones_like(seg_map) * 0.5, seg_map)
                final_map = torch.where(seg_hard[j,0] == 1, torch.ones_like(seg_map), disc_map)
                imageio.imsave(os.path.join(output_dir, name[:-3]+'jpg'), (final_map.cpu().numpy()*255).astype(np.uint8))
                output_dict['name'].append(name)
                output_dict['f1_score_cup'].append(dsc_cup.item())
                output_dict['f1_score_disc'].append(dsc_disc.item())
                output_dict['f1_score_avg'].append((dsc_cup.item() + dsc_disc.item()) / 2)
                print('Saving {}...'.format(name))

    pd.DataFrame.from_dict(output_dict).to_csv(os.path.join(output_dir, 'test_result.csv'), index=False)                


def visualization(train_loader, config, controller, output_dir):
    controller.cuda()
    controller.train()
    policies, _, _, _, _ = controller(4)
    parsed_policies = parse_policies(policies.cpu().detach().numpy(), config, logger=None)
    train_loader.dataset.transforms.transforms[0] = DGMultiPolicy(parsed_policies)

    for i, sample in enumerate(train_loader):
        input = sample['aug_images']
        for j, name in enumerate(sample['img_name']):
            for k in range(4):
                imageio.imsave(os.path.join(output_dir, name[:-4]+'_'+str(k)+'.jpg'), ((input[j*4+k].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8))
                print('Saving {}...'.format(os.path.join(output_dir, name[:-4]+'_'+str(k)+'.jpg')))
                print(sample['aug_policy'][j][k])


def test_rvs_augment_distribution(config, args):
    model, batch_size, workers = load_model(args, config)
    controller, M, _ = load_controller(args, config)
    train_samplers, train_loader, test_loader = get_seg_dg_dataloader(config, args, 4, workers)
    
    # load pretrained model
    if config.TEST.MODEL_DIR:
        try:
            model_state_file = os.path.join(config.TEST.MODEL_DIR, 'final_model_state.pth')
            model = utils.load_checkpoint(model_state_file, model)
        except:
            model_state_file = os.path.join(config.TEST.MODEL_DIR, 'final_state.pth')
            model = utils.load_checkpoint(model_state_file, model)
        
        model = model.cuda()
        print('Successfully loaded: {}'.format(model_state_file))
        if args.output_type == 'image':
            controller_state_file = os.path.join(config.TEST.MODEL_DIR, 'final_controller_state.pth')
            controller = utils.load_checkpoint(controller_state_file, controller)
            controller = controller.cuda()
            print('Successfully loaded: {}'.format(controller_state_file))

    # make output directory
    output_dir = Path(args.vis_dir)
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir()

    # save segmentation map
    if args.output_type == 'seg':
        inference(test_loader, model, output_dir, augment=False)
    # save augmentation image
    else:
        visualization(train_loader, config, controller, output_dir)


def test_optic_augment_distribution(config, args):
    model, batch_size, workers = load_model(args, config)
    controller, M, _ = load_controller(args, config)
    train_samplers, train_loader, test_loader = get_seg_dg_dataloader(config, args, 4, workers)
    
    # load pretrained model
    if config.TEST.MODEL_DIR:
        try:
            model_state_file = os.path.join(config.TEST.MODEL_DIR, 'model_best.pth')
            checkpoint = torch.load(model_state_file)
            state_dict = checkpoint.state_dict()
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module.' in k:
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
        except:
            try:
                model_state_file = os.path.join(config.TEST.MODEL_DIR, 'final_model_state.pth')
                model = utils.load_checkpoint(model_state_file, model)
            except:
                model_state_file = os.path.join(config.TEST.MODEL_DIR, 'final_state.pth')
                model = utils.load_checkpoint(model_state_file, model)

        model = model.cuda()
        print('Successfully loaded: {}'.format(model_state_file))
        if args.output_type == 'image':
            controller_state_file = os.path.join(config.TEST.MODEL_DIR, 'final_controller_state.pth')
            controller = utils.load_checkpoint(controller_state_file, controller)
            controller = controller.cuda()
            print('Successfully loaded: {}'.format(controller_state_file))

    # make output directory
    output_dir = Path(args.vis_dir)
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir()

    # save segmentation map
    if args.output_type == 'seg':
        optic_inference(test_loader, model, output_dir, augment=False)
    # save augmentation image
    else:
        visualization(train_loader, config, controller, output_dir)

def test_worker(config, args):
    args.distributed = False
    
    if config.DATASET.NAME in ['rvs']:
        test_rvs_augment_distribution(config, args)
    elif config.DATASET.NAME in ['optic']:
        test_optic_augment_distribution(config, args)
        