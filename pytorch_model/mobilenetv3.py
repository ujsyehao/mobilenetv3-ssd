'''MobileNetV3 in PyTorch.
'''
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from mmcv.cnn import (constant_init, kaiming_init, normal_init)
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES

def conv_bn(inp, oup, stride, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.output_status = False
        if kernel_size == 5 and in_size == 160 and expand_size == 672:
            self.output_status = True

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        if self.output_status:
            expand = out
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
	#Block(5, 160, 672, 160, hswish(), SeModule(160), 2)		
        if self.output_status:
            return (expand, out)
        return out

@BACKBONES.register_module
class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, ssd_body=True):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.use_body = ssd_body

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        #self.init_params()

        self.extra_convs = []
        if self.use_body:
	    # 1x1 256 -> 3x3 256 s=2 -> 1x1 512
            self.extra_convs.append(conv_1x1_bn(960, 256))
            self.extra_convs.append(conv_bn(256, 256, 2, groups=256))
            self.extra_convs.append(conv_1x1_bn(256, 512, groups=1))
	    # 1X1 128 -> 3X3 128 S=2 -> 1X1 256            
            self.extra_convs.append(conv_1x1_bn(512, 128))
            self.extra_convs.append(conv_bn(128, 128, 2, groups=128))
            self.extra_convs.append(conv_1x1_bn(128, 256))
	    # 1X1 128 -> 3X3 128 S=2 -> 1X1 256
            self.extra_convs.append(conv_1x1_bn(256, 128))
            self.extra_convs.append(conv_bn(128, 128, 2, groups=128))
            self.extra_convs.append(conv_1x1_bn(128, 256))
	    # 1X1 64 -> 3X3 64 S=2 -> 1X1 128
            self.extra_convs.append(conv_1x1_bn(256, 64))
            self.extra_convs.append(conv_bn(64, 64, 2, groups=64))
            self.extra_convs.append(conv_1x1_bn(64, 128))
            self.extra_convs = nn.Sequential(*self.extra_convs)
		
    def init_weights(self, pretrained=None):
        print (pretrained)
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        out = self.hs1(self.bn1(self.conv1(x)))

        #out = self.bneck(out)
        for i, block in enumerate(self.bneck):
            out = block(out)
            if isinstance(out, tuple):
                outs.append(out[0])
                out = out[1]

        out = self.hs2(self.bn2(self.conv2(out)))
				
        outs.append(out)

        for i, conv in enumerate(self.extra_convs):
            out = conv(out)
            if i % 3 == 2:
                outs.append(out)

        #print ('choose feature map nums: ')
        #print (len(outs))

        """
	if not self.use_body:
            out = F.avg_pool2d(out, 7)
            out = out.view(out.size(0), -1)
            out = self.hs3(self.bn3(self.linear3(out)))
            out = self.linear4(out)
        """
        return tuple(outs)

