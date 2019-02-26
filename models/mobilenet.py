"""MobileNet models

Based on:
    https://arxiv.org/abs/1704.04861
    https://arxiv.org/abs/1801.04381
"""

from collections import OrderedDict

import torch
from torch import nn


__all__ = ['mobilenetv1', 'mobilenetv2']


def build_DepthwiseSeparableConvolution(in_channels, out_channels,
                                        kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                  padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class BottleneckDWSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, expansion_factor=6, is_residual=True):
        super(BottleneckDWSConv2d, self).__init__()
        self.is_residual = is_residual
        if self.is_residual:
            assert stride == 1, 'Stride must be set to 1 in residual blocks'
            assert in_channels == out_channels, ('In and out number of '
                                                 'channels must match in '
                                                 'residual blocks.')
        internal_dim = in_channels * expansion_factor
        self.pointwiseConv = nn.Conv2d(in_channels, internal_dim,
                                       kernel_size=1, bias=False)
        self.pointwiseBN = nn.BatchNorm2d(internal_dim)
        self.pointwiseReLU = nn.ReLU6(inplace=True)
        self.dwiseConv = nn.Conv2d(internal_dim, internal_dim, kernel_size,
                                   stride=stride, padding=padding, bias=False,
                                   groups=internal_dim)
        self.dwiseBN = nn.BatchNorm2d(internal_dim)
        self.dwiseReLU = nn.ReLU6(inplace=True)
        self.linear = nn.Conv2d(internal_dim, out_channels, kernel_size=1,
                                bias=False)
        self.linearBN = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.pointwiseReLU(self.pointwiseBN(self.pointwiseConv(x)))
        h = self.dwiseReLU(self.dwiseBN(self.dwiseConv(h)))
        h = self.linearBN(self.linear(h))
        if self.is_residual:
            h += x
        return h


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.batch_dim = 1

    def forward(self, x):
        return x.flatten(self.batch_dim)


def mobilenetv1(pretrained=False, num_classes=1000):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)),
        ('dw_conv1', build_DepthwiseSeparableConvolution(32, 64, stride=1)),
        ('dw_conv2', build_DepthwiseSeparableConvolution(64, 128, stride=2)),
        ('dw_conv3', build_DepthwiseSeparableConvolution(128, 128, stride=1)),
        ('dw_conv4', build_DepthwiseSeparableConvolution(128, 256, stride=2)),
        ('dw_conv5', build_DepthwiseSeparableConvolution(256, 256, stride=1)),
        ('dw_conv6', build_DepthwiseSeparableConvolution(256, 512, stride=2)),
        ('dw_conv7', build_DepthwiseSeparableConvolution(512, 512, stride=1)),
        ('dw_conv8', build_DepthwiseSeparableConvolution(512, 512, stride=1)),
        ('dw_conv10', build_DepthwiseSeparableConvolution(512, 512, stride=1)),
        ('dw_conv11', build_DepthwiseSeparableConvolution(512, 512, stride=1)),
        ('dw_conv12', build_DepthwiseSeparableConvolution(512, 512, stride=1)),
        ('dw_conv13', build_DepthwiseSeparableConvolution(512, 1024, stride=2)),
        ('dw_conv14', build_DepthwiseSeparableConvolution(1024, 1024, stride=1)),
        ('classifier', nn.Sequential(OrderedDict([
            ('avg_pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('dropout', nn.Dropout2d()),
            ('linear', nn.Conv2d(1024, num_classes, 1)),
            ('flatten', Flatten()),
        ])))
    ]))


def mobilenetv2(pretrained=False, num_classes=1000):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )),
        ('bottleneck_1', BottleneckDWSConv2d(32, 16, stride=1,
                                             expansion_factor=1,
                                             is_residual=False)),
        ('bottleneck_2', nn.Sequential(
            BottleneckDWSConv2d(16, 24, stride=2, is_residual=False),
            BottleneckDWSConv2d(24, 24),
        )),
        ('bottleneck_3', nn.Sequential(
            BottleneckDWSConv2d(24, 32, stride=2, is_residual=False),
            BottleneckDWSConv2d(32, 32),
            BottleneckDWSConv2d(32, 32),
        )),
        ('bottleneck_4', nn.Sequential(
            BottleneckDWSConv2d(32, 64, stride=2, is_residual=False),
            BottleneckDWSConv2d(64, 64),
            BottleneckDWSConv2d(64, 64),
            BottleneckDWSConv2d(64, 64),
        )),
        ('bottleneck_5', nn.Sequential(
            BottleneckDWSConv2d(64, 96, stride=1, is_residual=False),
            BottleneckDWSConv2d(96, 96),
            BottleneckDWSConv2d(96, 96),
        )),
        ('bottleneck_6', nn.Sequential(
            BottleneckDWSConv2d(96, 160, stride=2, is_residual=False),
            BottleneckDWSConv2d(160, 160),
            BottleneckDWSConv2d(160, 160),
        )),
        ('bottleneck_7', BottleneckDWSConv2d(160, 320, is_residual=False)),
        ('conv2', nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )),
        ('classifier', nn.Sequential(OrderedDict([
            ('avg_pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('dropout', nn.Dropout2d()),
            ('flatten', Flatten()),
            ('linear', nn.Conv2d(1280, num_classes, 1)),
        ])))
    ]))
