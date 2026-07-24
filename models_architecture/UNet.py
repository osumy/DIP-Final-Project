import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False, norm='group'):
        super().__init__()
        if norm == 'instance':
            Norm = lambda c: nn.InstanceNorm2d(c, affine=True)
        elif norm == 'group':
            Norm = lambda c: nn.GroupNorm(32 if c >= 32 else 1, c)
        elif norm == 'batch':
            Norm = lambda c: nn.BatchNorm2d(c)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = Norm(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = Norm(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(0.2) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.relu1(self.n1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu2(self.n2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False, norm='group'):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, use_dropout, norm=norm)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.block(x)
        down = self.pool(feat)
        return feat, down


class Decoder(nn.Module):
    def __init__(self, up_in, skip_in, out_channels, use_dropout=False, norm='group'):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(up_in, out_channels, 3, padding=1, bias=False)
        )
        self.conv = ConvBlock(out_channels + skip_in, out_channels, use_dropout, norm=norm)

    def forward(self, x1, x2):
        x1_up = self.upconv(x1)
        if x1_up.size()[2:] != x2.size()[2:]:
            x1_up = F.interpolate(x1_up, size=x2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_dropout=False, norm='group', resize_output=False):
        super().__init__()
        self.resize_output = resize_output
        self.in_conv = ConvBlock(in_channels, 64, use_dropout, norm=norm)
        self.enc1 = Encoder(64, 128, use_dropout, norm=norm)
        self.enc2 = Encoder(128, 256, use_dropout, norm=norm)
        self.enc3 = Encoder(256, 512, use_dropout, norm=norm)
        self.enc4 = Encoder(512, 1024, use_dropout, norm=norm)

        self.dec1 = Decoder(1024, 512, 512, use_dropout, norm=norm)
        self.dec2 = Decoder(512, 256, 256, use_dropout, norm=norm)
        self.dec3 = Decoder(256, 128, 128, use_dropout, norm=norm)
        self.dec4 = Decoder(128, 64, 64, use_dropout, norm=norm)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, inp, target_size=None):
        x1 = self.in_conv(inp)
        f2, x2 = self.enc1(x1)
        f3, x3 = self.enc2(x2)
        f4, x4 = self.enc3(x3)
        f5, x5 = self.enc4(x4)

        x = self.dec1(x5, f4)
        x = self.dec2(x, f3)
        x = self.dec3(x, f2)
        x = self.dec4(x, x1)

        y = self.out_conv(x)
        y = inp + y
        if self.resize_output:
            target_size = target_size or x1.shape[2:]
            y = F.interpolate(y, size=target_size, mode='bilinear', align_corners=False)
        return y.clamp(-1, 1)