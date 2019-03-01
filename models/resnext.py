"""ResNeXt architecture.

Based on:
    https://arxiv.org/abs/1611.05431

Original implementation:
    https://github.com/facebookresearch/ResNeXt

This model is based on torchvision's ResNet model:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNeXt', 'resnext50_32x4d', 'resnext101_32x4d',
           'resnext101_64x4d', 'resnext152_32x4d']


model_urls = {
    'resnext50_32x4d': '',
    'resnext101_32x4d': '',
    'resnext101_64x4d': '',
    'resnext152_32x4d': '',
}


def group_conv3x3(in_channels, out_channels, groups, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        channels = cardinality * int(planes * base_width / 64)
        self.conv1 = conv1x1(inplanes, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = group_conv3x3(channels, channels, cardinality, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, cardinality, base_width,
                 num_classes=1000, zero_init_residual=False):
        super(ResNeXt, self).__init__()
        self.base_width = base_width
        self.cardinality = cardinality
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(block, 64, layers[0])
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn_expansion.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality,
                            self.base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality,
                                self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50-32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], 32, 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnext50_32x4d']))
    return model


def resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNext-101-32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], 32, 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnext101_32x4d']))
    return model


def resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNext-101-64x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], 64, 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnext101_64x4d']))
    return model


def resnext152_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-152-32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], 32, 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnext152_32x4d']))
    return model
