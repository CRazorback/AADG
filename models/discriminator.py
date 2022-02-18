import torch
import torch.nn as nn


class FeatureDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dis = nn.Sequential(nn.Linear(1280, 128),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(128, num_classes))

    def forward(self, x):
        # x = self.avgpool(x).squeeze()
        x = self.dis(x)

        return x


class MomentumFeatureDiscriminator(nn.Module):
    def __init__(self, num_classes, in_channels, m=0.999):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.m = m
        self.dis = nn.Sequential(nn.Linear(in_channels, 128),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Linear(128, num_classes)                       
        self.mom_dis = nn.Sequential(nn.Linear(in_channels, 128),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.mom_fc = nn.Linear(128, num_classes)                               

    def momentum_update(self):
        for param_q, param_k in zip(self.dis.parameters(), self.mom_dis.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.fc.parameters(), self.mom_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def synchronize_parameters(self):
        for param_q, param_k in zip(self.dis.parameters(), self.mom_dis.parameters()):
            param_k.data = param_q.data

        for param_q, param_k in zip(self.fc.parameters(), self.mom_fc.parameters()):
            param_k.data = param_q.data

    def forward(self, x, momentum=False, return_feature=False):
        # x = self.avgpool(x).squeeze()
        if momentum:
            with torch.no_grad():
                fe = self.mom_dis(x)
                x = self.mom_fc(fe)
        else:
            fe = self.dis(x)
            x = self.fc(fe)

        if return_feature:
            return x, fe
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.f = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        return self.f(x)
        

class ImageDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.f1 = ConvBlock(3, 32)
        self.f2 = ConvBlock(32, 64)
        self.f3 = ConvBlock(64, 128)
        self.f4 = ConvBlock(128, 256)
        self.f5 = ConvBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dis = nn.Sequential(nn.Linear(512, 128),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_feature=False):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.avgpool(x).squeeze()
        fe = self.dis(x)
        x = self.fc(fe)

        if return_feature:
            return x, fe
        else:
            return x
