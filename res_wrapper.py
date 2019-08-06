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
                 short_len=1,
                 long_len=16,
                 new_length=1,
                 dropout=0.5,
                 n_class=174):
        print('n_layer:', n_layer)
        if n_layer == 34:
            super(ResWrapper, self).__init__(BasicBlock, [3, 4, 6, 3], n_class,
                                             dropout)
            self.load_pretrained_weights('resnet34')
        elif n_layer == 50:
            super(ResWrapper, self).__init__(Bottleneck, [3, 4, 6, 3], n_class,
                                             dropout)
            self.load_pretrained_weights('resnet50')
        else:
            raise KeyError('Not implemented layer={}'.format(n_layer))
        self.modality = modality
        self.short_len = short_len
        self.long_len = long_len
        self.new_length = new_length

        self.to_rank_1 = nn.Conv2d(512 * 2, 512, kernel_size=1, padding=0)
        self.to_rank_2 = nn.Conv2d(512 * 2, 512, kernel_size=1, padding=0)

        self.to_2rank_1 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.to_2rank_2 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)

        self.to_3rank_1 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.to_3rank_2 = nn.Conv2d(2 * 512, 512, kernel_size=1, padding=0)
        self.abilip_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, n_class)

    def forward(self, x):
        if self.modality == 'RGB':
            x = x.view((-1, self.long_len, self.short_len,
                        self.new_length * 3) + x.size()[-2:])
        else:
            x = x.view((-1, self.long_len, self.short_len,
                        self.new_length * 2) + x.size()[-2:])

        (b, long_len, short_len, ch, h, w) = x.size()
        # print('start x.size:', x.size())

        x = x.view(b * long_len * short_len, ch, h, w)
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
        n_feat = x.size(1)
        x = x.view(b, short_len, n_feat, 7, 7)
        # print('before minus: x.size:', x.size())
        x = (x[:, :-1, ...] - x[:, 1:, ...]).contiguous()
        x = F.normalize(x, p=2, dim=2).contiguous()
        # print('after minus: x.size:', x.size())
        x = x.view(b * 4, (short_len - 1) // 4 * n_feat, 7, 7)
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
        x = x.view(b * 2, 2 * 512, 7, 7)
        x1 = self.to_2rank_1(x)
        # print('after to_2rank_1 x1.size:', x1.size())
        x = self.to_2rank_2(x)
        # print('after to_2rank_2 x.size:', x.size())
        x = x1 * x
        # x = self.top_b2(x)

        # === third layer
        x = x.view(b, 2 * 512, 7, 7)
        x1 = self.to_3rank_1(x)
        # print('after to_3rank_1 x1.size:', x1.size())
        x = self.to_3rank_2(x)
        # print('after to_3rank_2 x.size:', x.size())
        x = x1 * x

        # === classification
        x = self.abilip_pool(x)
        # print('before fc x.size:', x.size())
        x = x.view(b, 512)
        logits = self.fc2(self.dropout(x))
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
