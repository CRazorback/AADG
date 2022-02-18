import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resize=None):
        super().__init__()
        self.resize = resize

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if self.resize == 'up':
            self.f = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if self.resize == 'down':
            self.f = nn.MaxPool2d((2, 2))

    def forward(self, x1, x2=None):
        if self.resize: x1 = self.f(x1)
        if self.resize == 'up': 
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x1, x2], axis=1)
        x = self.conv(x1)      
        
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.fe = ConvBlock(self.n_channels, 32)
        self.d1 = ConvBlock(32, 64, 'down')
        self.d2 = ConvBlock(64, 128, 'down')
        self.d3 = ConvBlock(128, 256, 'down')
        self.d4 = ConvBlock(256, 512, 'down')
        self.d5 = ConvBlock(512, 512, 'down')
        self.u1 = ConvBlock(512 + 512, 512, 'up')
        self.u2 = ConvBlock(512 + 256, 256, 'up')
        self.u3 = ConvBlock(256 + 128, 128, 'up')
        self.u4 = ConvBlock(128 + 64, 64, 'up')
        self.u5 = ConvBlock(64 + 32, 32, 'up')
        self.pred = nn.Conv2d(32, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.fe(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x6 = self.d5(x5)
        x7 = self.u1(x6, x5)
        x8 = self.u2(x7, x4)
        x9 = self.u3(x8, x3)
        x10 = self.u4(x9, x2)
        x11 = self.u5(x10, x1)
        logits = self.pred(x11)

        return logits, x6