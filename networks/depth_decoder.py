# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .non_local_embedded_gaussian import NonLocalBlock2D


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=80, use_skips=True, min_depth=0.1, max_depth=100., model_type='base'):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        self.min_depth = max(1.0, min_depth)
        self.max_depth = max_depth
        self.model_type = model_type
        


        # decoder
        self.convs = OrderedDict()
        if self.model_type == 'nlb' or self.model_type == 'full':
            self.convs["nlb"] = NonLocalBlock2D(self.num_ch_enc[-1], inter_channels=self.num_ch_enc[-1], sub_sample=False)
        for i in range(4, -1, -1):
            # upconv_0
            if self.model_type == 'ddv' or self.model_type == 'full':
                if i == 4:
                    num_ch_in = self.num_ch_enc[-1] 
                elif i == 3:
                    num_ch_in = self.num_ch_dec[i + 1]
                else:
                    num_ch_in = self.num_output_channels
            else:
                num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            
            
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        if self.model_type == 'ddv' or self.model_type == 'full':
            for s in self.scales:
                self.convs[("dpv", s)] = ConvBlock(self.num_ch_dec[s], self.num_output_channels)
            #self.depth_linsp = [np.exp(np.log(self.min_depth) + np.log(self.max_depth / self.min_depth) * i / self.num_output_channels) for i in range(self.num_output_channels)]
            #self.disp_linsp = (1 / torch.tensor(self.depth_linsp).cuda() - 1 / self.max_depth) / (1 / self.min_depth - 1 / self.max_depth)
            self.depth_linsp = [min(1.0, min_depth)]
            self.depth_linsp += [np.exp(np.log(self.min_depth) + np.log(self.max_depth / self.min_depth) * i / (self.num_output_channels - 2)) for i in range(self.num_output_channels - 1)]
            self.disp_linsp = (1 / torch.tensor(self.depth_linsp).cuda() - 1 / self.max_depth) / (1 / self.min_depth - 1 / self.max_depth)
        
        else:
            self.num_output_channels = 1
            for s in self.scales:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        if self.model_type == 'nlb' or self.model_type == 'full':
            x = self.convs["nlb"](x)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if self.model_type == 'ddv' or self.model_type == 'full':
                    x = self.convs[("dpv", i)](x)
                    self.outputs[("disp", i)] = (F.softmax(x, dim=1).permute(0, 2, 3, 1) * self.disp_linsp).permute(0, 3, 1, 2).sum(dim=1, keepdim=True)
                else:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
