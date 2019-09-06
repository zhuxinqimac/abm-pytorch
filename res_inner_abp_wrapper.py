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

from res_inner_abp import res_IABP, BasicBlock, BasicBlockIABP, Bottleneck
from res_inner_abp import BottleneckIABP

class ResIABPWrapper(res_IABP):
    def __init__(self, modality='RGB', n_layer=34, short_len=8, long_len=1, 
            new_length=1, dropout=0.5, n_class=174, dataset='something'):
        print('n_layer:', n_layer)
        if n_layer == 34:
            super(ResIABPWrapper, self).__init__(BasicBlock, BasicBlockIABP, 
                    [3, 4, 6, 3], n_class, dropout, short_len)
            self.load_pretrained_weights('resnet34')
            self.expansion = 1
        elif n_layer == 50:
            super(ResIABPWrapper, self).__init__(Bottleneck, BottleneckIABP, 
                    [3, 4, 6, 3], n_class, dropout, short_len)
            self.load_pretrained_weights('resnet50')
            self.expansion = 4
        else:
            raise KeyError('Not implemented layer={}'.format(n_layer))

        self.modality = modality
        self.short_len = short_len
        self.long_len = long_len
        self.new_length = new_length
        self.dataset = dataset

        if (modality == 'RGB') or (self.dataset == 'something_v2'):
            self.single_frame_channel = 3
        else:
            self.single_frame_channel = 2

        if (self.modality == 'Flow') or \
                not(self.single_frame_channel*self.new_length==3):
            self._reconstruct_first_layer()

        self.abilip_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512*self.expansion, n_class)

    def _reconstruct_first_layer(self):
        print('Reconstructing first conv...')
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], 
            nn.Conv2d), list(range(len(modules)))))[0]
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
            new_kernels = params[0].data.mean(dim=1, 
                    keepdim=True).expand(new_kernel_size).contiguous()
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
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

    def forward(self, x):
        if (self.modality == 'RGB') or (self.dataset == 'something_v2'):
            x = x.view((-1, self.short_len, 
                self.new_length*3) + x.size()[-2:])
        else:
            x = x.view((-1, self.short_len, 
                self.new_length*2) + x.size()[-2:])

        (b, short_len, ch, h, w) = x.size()
        # print('start x.size:', x.size())

        x = x.view(b*short_len, ch, h, w)
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
        
        x = self.abilip_pool(x)
        # x = x.view(b, short_len//2, -1)
        x = x.view(b, short_len, -1)
        # print('after pool: x.size:', x.size())
        x = x.mean(1).view(b, -1)
        # print('after mean: x.size:', x.size())
        x = self.dropout(x)
        logits = self.fc2(x)
        # pdb.set_trace()
        return logits

    def get_optim_policies(self):
        trainable = list(self.parameters())
        return [{'params': trainable, 'lr_mult': 1, 'decay_mult': 1, 
                'name': "trainable parameters"}]
