import os
import time
import json
import shutil
import pdb
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm

from res_single_frame import res_SF, BasicBlock, Bottleneck


class ResWrapper(res_SF):

    def __init__(self,
                 modality='RGB',
                 n_layer=34,
                 short_len=16,
                 long_len=1,
                 new_length=1,
                 dropout=0.5,
                 n_class=174,
                 dataset='something'):
        print('n_layer:', n_layer)
        if n_layer == 34:
            super(ResWrapper, self).__init__(BasicBlock, [3, 4, 6, 3], n_class,
                                             dropout)
            self.load_pretrained_weights('resnet34')
        else:
            raise KeyError('Not implemented layer={}'.format(n_layer))

        self.modality = modality
        self.short_len = short_len
        self.long_len = long_len
        self.new_length = new_length

        if (modality == 'RGB') or (self.dataset == 'something_v2'):
            self.single_frame_channel = 3
        else:
            self.single_frame_channel = 2

        if (self.modality == 'Flow') or \
                not(self.single_frame_channel*self.new_length==3):
            self._reconstruct_first_layer()

        self.to_rank_1 = nn.Conv2d(512 * short_len // 4,
                                   512,
                                   kernel_size=1,
                                   padding=0)
        self.to_rank_2 = nn.Conv2d(512 * short_len // 4,
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
            self.cls = nn.Linear(512, n_class)
        else:
            self.cls = nn.Sequential(nn.Dropout(p=dropout),
                                     nn.Linear(512, n_class))

    def _reconstruct_first_layer(self):
        print('Reconstructing first conv...')
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.modules())
        first_conv_idx = list(
            filter(lambda x: isinstance(modules[x], nn.Conv2d),
                   list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.single_frame_channel * \
                self.new_length, ) + \
                kernel_size[2:]
        # if not(self.single_frame_channel == 3):
        if self.modality == 'Flow':
            new_kernels = params[0].data.mean(
                dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernels = params[0].data.repeat([1,self.new_length,]+\
                    [1]*(len(kernel_size[2:]))).contiguous()

        new_conv = nn.Conv2d(self.single_frame_channel * \
                self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride,
                             conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys(
        ))[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

    def forward(self, x):
        if (self.modality == 'RGB') or (self.dataset == 'something_v2'):
            x = x.view((-1, self.short_len, self.new_length * 3) +
                       x.size()[-2:])
        else:
            x = x.view((-1, self.short_len, self.new_length * 2) +
                       x.size()[-2:])

        (b, short_len, ch, h, w) = x.size()
        # print('start x.size:', x.size())

        x = x.view(b * short_len, ch, h, w)
        # print('before resnet x.size:', x.size())

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

        # ABiliP >>>
        _, n_feat, height, width = x.size()
        x = x.view(b * 4, short_len // 4 * n_feat, height, width)
        # print('before to_rank x.size:', x.size())
        # x size: (batch*4, (D//4)*n_feat, 7, 7)
        x1 = self.to_rank_1(x)
        # print('after to_rank_1 x1.size:', x1.size())
        # x1.size: (batch*4, 512, 7, 7)
        x = self.to_rank_2(x)
        # print('after to_rank_2 x2.size:', x.size())
        # x2.size: (batch*4, 512, 7, 7)
        x = x1 * x
        # x = self.top_b1(x)

        # === second layer
        x = x.view(b * 2, 2 * 512, height, width)
        x1 = self.to_2rank_1(x)
        # print('after to_2rank_1 x1.size:', x1.size())
        x = self.to_2rank_2(x)
        # print('after to_2rank_2 x.size:', x.size())
        x = x1 * x
        # x = self.top_b2(x)

        # === third layer
        x = x.view(b, 2 * 512, height, width)
        x1 = self.to_3rank_1(x)
        # print('after to_3rank_1 x1.size:', x1.size())
        x = self.to_3rank_2(x)
        # print('after to_3rank_2 x.size:', x.size())
        x = x1 * x

        # === classification
        x = self.post_avgpool(x)
        x = x.view((x.size()[0], -1))

        logits = self.cls(x)
        # ABiliP <<<

        # pdb.set_trace()
        return logits

    def get_optim_policies(self):
        trainable = list(self.parameters())
        return [{
            'params': trainable,
            'lr_mult': 1,
            'decay_mult': 1,
            'name': "trainable parameters"
        }]

    # def get_optim_policies(self):
    # self.key_names = [
    # 'to_rank_1', 'to_rank_2', 'top_b1', 'to_2rank_1', 'to_2rank_2',
    # 'top_b2', 'to_3rank_1', 'to_3rank_2', 'fc2'
    # ]
    # general_parameters = []
    # key_parameters = []
    # for k, v in self.named_parameters():
    # if k.split('.')[0] in self.key_names:
    # key_parameters.append(v)
    # else:
    # general_parameters.append(v)
    # parameters = []
    # parameters.append({
    # 'params': general_parameters,
    # 'lr_mult': 1.0,
    # 'decay_mult': 1.,
    # 'name': 'general_params'
    # })
    # parameters.append({
    # 'params': key_parameters,
    # 'lr_mult': 1.0,
    # 'decay_mult': 1.,
    # 'name': 'key_params'
    # })
    # return parameters
