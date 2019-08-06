import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['res_IABP', 'res_iabp_18', 'res_iabp_34', \
        'res_iabp_50', 'res_iabp_101', 'res_iabp_152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv3x3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockIABP(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, tlen=8):
        super(BasicBlockIABP, self).__init__()

        self.tlen = tlen
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=False)
        self.conv1_abp = conv3x3x3(inplanes, planes, stride, bias=True)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # abp origin branch >>>
        (_, dim, h, w) = x.size()
        x = x.view(-1, self.tlen, dim, h, w)
        b = x.size(0)
        # print('in basicabp x.size:', x.size())
        x = x.transpose(1, 2).contiguous()

        out = self.conv1(x)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b * self.tlen, dim, h, w)
        out = self.bn1(out)
        # abp origin branch <<<

        # abp added branch >>>
        out_abp = self.conv1_abp(x)
        out_abp = out_abp.transpose(1, 2).contiguous()
        out_abp = out_abp.view(b * self.tlen, dim, h, w)
        out = out * out_abp
        # abp added branch <<<

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckIABP(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, tlen=8):
        super(BottleneckIABP, self).__init__()
        self.tlen = tlen

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.conv2_abp = nn.Conv3d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # abp origin branch >>>
        (_, dim, h, w) = out.size()
        out1 = out.view(-1, self.tlen, dim, h, w)
        b = out1.size(0)
        # print('in bottleblockabp out1.size:', out1.size())
        out1 = out1.transpose(1, 2).contiguous()

        out = self.conv2(out1)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b * self.tlen, dim, h, w)
        out = self.bn2(out)
        # abp origin branch <<<

        # abp added branch >>>
        out_abp = self.conv2_abp(out1)
        out_abp = out_abp.transpose(1, 2).contiguous()
        out_abp = out_abp.view(b * self.tlen, dim, h, w)
        out = out * out_abp
        # abp added branch <<<

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class res_IABP(nn.Module):

    def __init__(self,
                 block,
                 abpblock,
                 layers,
                 num_classes=1000,
                 dropout=0,
                 tlen=8,
                 **kwargs):
        self.inplanes = 64
        self.tlen = tlen
        super(res_IABP, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                     stride=(2, 1, 1),
                                     padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.layer1 = self._make_layer_abp(block, abpblock,
        # 64, layers[0], tlen=self.tlen)
        self.layer2 = self._make_layer_abp(block,
                                           abpblock,
                                           128,
                                           layers[1],
                                           stride=2,
                                           tlen=self.tlen)
        self.layer3 = self._make_layer_abp(block,
                                           abpblock,
                                           256,
                                           layers[2],
                                           stride=2,
                                           tlen=self.tlen)
        self.layer4 = self._make_layer_abp(block,
                                           abpblock,
                                           512,
                                           layers[3],
                                           stride=2,
                                           tlen=self.tlen)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_abp(self, block, abpblock, planes, blocks, stride=1,
                        tlen=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(abpblock(self.inplanes, planes, tlen=tlen))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        x: (batch, n_channel, h, w)
        '''
        # print('x.size:', x.size())
        x = self.conv1(x)
        # print('after con1.size:', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('after maxpool1.size:', x.size())

        x = self.layer1(x)
        # print('after layer1.size:', x.size())
        x = self.layer2(x)
        # print('after layer2.size:', x.size())
        x = self.layer3(x)
        # print('after layer3.size:', x.size())
        x = self.layer4(x)
        # print('after layer4.size:', x.size())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)

        # input('oo')
        return x

    def load_pretrained_weights(self, imagenet_name=None):
        if imagenet_name:
            ckpt_dict = model_zoo.load_url(model_urls[imagenet_name])
            model_dict = self.state_dict()
            for k in model_dict.keys():
                if k in ckpt_dict:
                    if ckpt_dict[k].size() == model_dict[k].size():
                        model_dict[k] = ckpt_dict[k]
                    else:
                        print('orgin to 3d, k=', k)
                        with torch.no_grad():
                            model_dict[k].zero_()
                            model_dict[k][:, :, 1] = ckpt_dict[k]
                else:
                    if 'abp' in k:
                        k_type = k.split('.')[-1]
                        print('added_abp 3d, k=', k)
                        with torch.no_grad():
                            if k_type == 'weight':
                                model_dict[k].zero_()
                            elif k_type == 'bias':
                                model_dict[k] = model_dict[k].zero_() + 1
                            else:
                                print('!! unexpected k_type: k=', k)
                    else:
                        print('random param: k=', k)
            self.load_state_dict(model_dict)
            print('=> loaded imagenet weights.')
        else:
            print('!! not using imagenet weights.')


def res_iabp_18(imagenet_name=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet18'
    else:
        imagenet_name = None
    model = res_IABP(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_iabp_34(imagenet_name=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet34'
    else:
        imagenet_name = None
    model = res_IABP(BasicBlock, [3, 4, 6, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_iabp_50(imagenet_name=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet50'
    else:
        imagenet_name = None
    model = res_IABP(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_iabp_101(imagenet_name=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet101'
    else:
        imagenet_name = None
    model = res_IABP(Bottleneck, [3, 4, 23, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_iabp_152(imagenet_name=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet152'
    else:
        imagenet_name = None
    model = res_IABP(Bottleneck, [3, 8, 36, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model
