# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Net2(nn.Module):
    """
    Fully Connected Network
    """

    def __init__(self, opt):
        super(Net2, self).__init__()
        fc_nums = [len(opt.input_size)] + opt.fc_nums
        # 全连接层
        self.fc_layers = nn.ModuleList([nn.Linear(fc_nums[i-1], fc_nums[i]) for i, fc in enumerate(fc_nums[:-1], 1)])
        self.drop_out = nn.Dropout(opt.dropout)

        self.reset_para()

    def reset_para(self):
        '''
        重新初始化参数
        '''
        for fc in self.fc_layers:
            nn.init.uniform_(fc.weight, -0.2, 0.2)
            nn.init.constant_(fc.bias, 0.1)

    def forward(self, x):
        input_x = x.float()

        # 经过线性层
        for fc in self.fc_layers[:-1]:
            input_x = fc(input_x).relu()
            input_x = self.drop_out(input_x)

        x = self.fc_layers[-1](input_x)
        return x
