from res_single_frame import res_SF, Bottleneck, BasicBlock
import torch.nn.functional as F
import torch.nn as nn
import torch

__all__ = ['res_AAMFB', 'res_aamfb_18', 'res_aamfb_34', \
        'res_aamfb_50', 'res_aamfb_101', 'res_aamfb_152']


class res_AAMFB(res_SF):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 dropout=0,
                 n_cframes=16,
                 **kwargs):
        super(res_AAMFB, self).__init__(block, layers, num_classes, dropout)
        self.to_rank_1 = nn.Conv2d((n_cframes // 4) * 512 * block.expansion,
                                   512,
                                   kernel_size=1,
                                   padding=0)
        self.to_rank_2 = nn.Conv2d((n_cframes // 4) * 512 * block.expansion,
                                   512,
                                   kernel_size=1,
                                   padding=0)

        self.to_2rank_1 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.to_2rank_2 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)

        self.to_3rank_1 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.to_3rank_2 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.post_avgpool = nn.AdaptiveAvgPool2d(1)
        print('dropout == ', str(dropout))
        if dropout == 0:
            self.cls = nn.Linear(512, num_classes)
        else:
            self.cls = nn.Sequential(nn.Dropout(p=dropout),
                                     nn.Linear(512, num_classes))

    def forward(self, x):
        '''
        x: (batch, n_channel, d, h, w)
        '''
        x_size = x.size()
        # print('x.size:', x.size())
        x = x.transpose(1, 2).contiguous().view((-1, x_size[1]) + x_size[-2:])
        # print('after reshape x.size:', x.size())
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
        # x size: (batch*D, n_feat, 7, 7)

        # === temporal relation model along time
        len_feat = x.size()[1]
        height, width = x.size()[-2], x.size()[-1]
        x = x.view((x_size[0] * 4, (x_size[2] // 4) * len_feat, height, width))
        # x size: (batch*4, (D//4)*n_feat, 7, 7)
        # print('before to_rank x.size:', x.size())
        x1 = self.to_rank_1(x)
        # print('after to_rank_1 x1.size:', x1.size())
        # x1.size: (batch*4, 512, 7, 7)
        x = self.to_rank_2(x)
        # print('after to_rank_2 x2.size:', x.size())
        # x2.size: (batch*4, 512, 7, 7)
        x = x1 * x

        # === second layer
        x = x.view((x_size[0] * 2, 2 * 512, height, width))
        x1 = self.to_2rank_1(x)
        # print('after to_2rank_1 x1.size:', x1.size())
        x = self.to_2rank_2(x)
        # print('after to_2rank_2 x.size:', x.size())
        x = x1 * x

        # === third layer
        x = x.view((x_size[0], 2 * 512, height, width))
        x1 = self.to_3rank_1(x)
        # print('after to_3rank_1 x1.size:', x1.size())
        x = self.to_3rank_2(x)
        # print('after to_3rank_2 x.size:', x.size())
        x = x1 * x

        x = self.post_avgpool(x)
        x = x.view((x.size()[0], -1))

        x = self.cls(x)
        return x


def res_aamfb_18(imagenet_name=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet18'
    else:
        imagenet_name = None
    model = res_AAMFB(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_aamfb_34(imagenet_name=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet34'
    else:
        imagenet_name = None
    model = res_AAMFB(BasicBlock, [3, 4, 6, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_aamfb_50(imagenet_name=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet50'
    else:
        imagenet_name = None
    model = res_AAMFB(Bottleneck, [3, 4, 6, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_aamfb_101(imagenet_name=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet101'
    else:
        imagenet_name = None
    model = res_AAMFB(Bottleneck, [3, 4, 23, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model


def res_aamfb_152(imagenet_name=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if imagenet_name:
        imagenet_name = 'resnet152'
    else:
        imagenet_name = None
    model = res_AAMFB(Bottleneck, [3, 8, 36, 3], **kwargs)
    model.load_pretrained_weights(imagenet_name)
    return model
