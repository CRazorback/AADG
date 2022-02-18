import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .backbone import resnet, xception, drn, mobilenet


def build_decoder(num_classes, num_domain, backbone, BatchNorm):
    return Decoder(num_classes, num_domain, backbone, BatchNorm)


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)


def build_encoderDC(Num_D, backbone, BatchNorm):
    return EncoderDC(Num_D, backbone, BatchNorm)


class EncoderDC(nn.Module):
    def __init__(self, Num_D, backbone, BatchNorm):
        super(EncoderDC, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        inplanes = 256
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.cls = nn.Conv2d(inplanes, Num_D, 1)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls(x)

        return torch.squeeze(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        feature = x
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x), feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, num_domain, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn = BatchNorm(304)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, feature, low_level_feat, domain_code, centroids):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat_ = low_level_feat
        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(feature, size=low_level_feat_.size()[2:], mode='bilinear', align_corners=True)
        feature = torch.cat((x, low_level_feat_), dim=1)
        feature = self.bn(feature)
        x = feature
        # feature = self.relu(feature)
        # x, hal_scale, sel_scale = self.embedding(feature, domain_code, centroids)
        hal_scale, sel_scale = None, None

        x = self.last_conv(x)

        return x, feature, hal_scale, sel_scale

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, num_domain=3, freeze_bn=False, lam=0.9):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.lam = lam
        self.centroids = nn.Parameter(torch.randn(num_domain, 304, 64, 64), requires_grad=False)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, num_domain, backbone, BatchNorm)
        self.last_conv_mask = nn.Sequential(BatchNorm(3),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Conv2d(3, num_domain, kernel_size=1, stride=1))
        # build encoder for domain code
        self.encoder_d = build_encoderDC(num_domain, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def update_memory(self, feature):
        _feature = torch.mean(torch.mean(feature, 3, True), 2, True)
        lam = self.lam
        self.centroids[0].data = lam * self.centroids[0].data + (1 - lam) * torch.mean(_feature[0:8], 0, True)
        self.centroids[1].data = lam * self.centroids[1].data + (1 - lam) * torch.mean(_feature[8:16], 0, True)
        self.centroids[2].data = lam * self.centroids[2].data + (1 - lam) * torch.mean(_feature[16:24], 0, True)

    def forward(self, input, extract_feature=False):
        x, low_level_feat = self.backbone(input)

        x, feature = self.aspp(x)
        domain_code = self.encoder_d(x)
        x, feature, hal_scale, sel_scale = self.decoder(x, feature, low_level_feat, domain_code, self.centroids)

        if extract_feature:
            return feature

        # torch.mean(torch.mean(centroids, 3, True), 2, True)
        self.update_memory(feature)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_para(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # for param in self.aspp.parameters():
        #     param.requires_grad = False


    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
