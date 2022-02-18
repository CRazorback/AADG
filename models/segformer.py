import torch
import torch.nn as nn

from mmseg.models.backbones.mix_transformer import mit_b1, mit_b2, mit_b3
from mmseg.models.decode_heads import segformer_head
from mmcv.runner import load_checkpoint


class SegFormer(nn.Module):
    def __init__(self, cfg, num_classes) -> None:
        super().__init__()
        checkpoint = cfg.MODEL.PRETRAINED_WEIGHTS
        if 'mit_b1' in checkpoint:
            self.backbone = mit_b1()
            self.head = segformer_head.SegFormerHead(
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=256),
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        elif 'mit_b2' in checkpoint:
            self.backbone = mit_b2()
            self.head = segformer_head.SegFormerHead(
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=768),
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        elif 'mit_b3' in checkpoint:
            self.backbone = mit_b3()
            self.head = segformer_head.SegFormerHead(
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=768),
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        else:
            raise ValueError('unknown backbone')

        if 'segformer' in checkpoint:
            state_dict = torch.load(checkpoint)
            backbone_state_dict = {}
            head_state_dict = {}
            for k, v in state_dict['state_dict'].items():
                if 'conv_seg' in k or 'linear_pred' in k:
                    continue
                if 'backbone.' in k:
                    backbone_state_dict[k[9:]] = v
                if 'decode_head.' in k:
                    head_state_dict[k[12:]] = v

            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            self.head.load_state_dict(head_state_dict, strict=False)
        else:
            self.backbone.init_weights(checkpoint)
        
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=4.0)

    def forward(self, x):
        fe = self.backbone(x)
        pred = self.head(fe)
        pred = self.upsampling(pred)

        return pred
