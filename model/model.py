import torch
import numpy as np
from model import build_backbone


class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size,
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, network, datasets, view=True):
        super(parsingNet, self).__init__()
        self.w = datasets['input_size'][0]
        self.h = datasets['input_size'][1]
        self.cls_dim = (datasets['griding_num'] + 1, datasets['num_per_lane'], datasets['num_lanes'])# (num_gridding, num_cls_per_lane, num_of_lanes) (201, 18, 4)
        # num_cls_per_lane is the number of row anchors
        self.use_aux = datasets['use_aux']
        self.total_dim = np.prod(self.cls_dim)
        self.c1, self.c2, self.c3 = network['out_channel']
        # input : nchw,
        # output: (w+1) * sample_rows * 4  14472
        self.model = build_backbone(network['backbone'])
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(self.c1, self.c1, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(self.c1, self.c1,3,padding=1),
                conv_bn_relu(self.c1, self.c1,3,padding=1),
                conv_bn_relu(self.c1, self.c1,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(self.c2, self.c1, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(self.c1, self.c1,3,padding=1),
                conv_bn_relu(self.c1, self.c1,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(self.c3, self.c1, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(self.c1,self.c1,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(self.c1*3, self.c2, 3,padding=2,dilation=2),
                conv_bn_relu(self.c2, self.c1, 3,padding=2,dilation=2),
                conv_bn_relu(self.c1, self.c1, 3,padding=2,dilation=2),
                conv_bn_relu(self.c1, self.c1, 3,padding=4,dilation=4),
                torch.nn.Conv2d(self.c1, self.cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(8*self.w//32* self.h//32, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(self.c3,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        initialize_weights(self.cls)

    def forward(self, x):
        x2,x3,fea = self.model(x)
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None
        fea = self.pool(fea).view(-1, 8*self.w//32* self.h//32)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        if self.use_aux:
            return group_cls, aux_seg
        return group_cls


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)