import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None,sub_sample=False, bn_layer=True, only_res=True):
        """
        :param in_channels:
        :param inter_channels:
        :param sub_sample:
        """
        super(NonLocalBlock2D, self).__init__()

        self.sub_sample = sub_sample
        self.only_res = only_res
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        
        if not self.only_res:
            if bn_layer:
                self.W = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0),
                    bn(self.in_channels)
                )
                nn.init.constant_(self.W[1].weight, 0)
                nn.init.constant_(self.W[1].bias, 0)
            else:
                self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                                 kernel_size=1, stride=1, padding=0)
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)


        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0) # b

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # [b, c1, h, w] -> [b, c1, h*w] or [b, c1, h/2, w/2] -> [b, c1, h*w/4]
        g_x = g_x.permute(0, 2, 1) # [b, h*w, c1] or [b, h*w/4, c1]

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # [b, c1, h, w] -> [b, c1, h*w]
        theta_x = theta_x.permute(0, 2, 1) #[b, h*w, c1]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # [b, c1, h, w] -> [b, c1, h*w] or [b, c1, h/2, w/2] -> [b, c1, h*w/4]
        f = torch.matmul(theta_x, phi_x) # [b, h*w, h*w] or [b, h*w, h*w/4]
        f_div_C = F.softmax(f, dim=-1) # [b, h*w, h*w] or [b, h*w, h*w/4]

        y = torch.matmul(f_div_C, g_x) # [b, h*w, c1]
        y = y.permute(0, 2, 1).contiguous() # [b, c1, h*w]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # [b, c1, h, w]
        if not self.only_res:
            W_y = self.W(y)
            z = W_y + x
        else:
            z = y

        if return_nl_map:
            return z, f_div_C
        return z