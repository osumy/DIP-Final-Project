import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, c, norm='group'):
        super().__init__()
        Norm = {'instance': nn.InstanceNorm2d,
                'batch': nn.BatchNorm2d,
                'group': lambda ch: nn.GroupNorm(32 if ch >= 32 else 1, ch)}[norm]
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, bias=False),
            Norm(c),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, bias=False),
            Norm(c),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_resnet_blocks=9, norm='group'):
        super().__init__()
        Norm = {'instance': nn.InstanceNorm2d,
                'batch': nn.BatchNorm2d,
                'group': lambda ch: nn.GroupNorm(32 if ch >= 32 else 1, ch)}[norm]

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, bias=False),
            Norm(64),
            nn.ReLU(inplace=True)
        ]

        in_ch = 64
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                Norm(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        for _ in range(num_resnet_blocks):
            layers += [ResBlock(in_ch, norm=norm)]

        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                Norm(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, bias=True)
        ]
        self.body = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = x + self.body(x)
        return y.clamp(-1, 1)