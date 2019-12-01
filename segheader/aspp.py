import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.conv_id = 0
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding[self.conv_id], dilation=dilation[self.conv_id], bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        for _ in self.dilation:
            x = self.atrous_conv(x)
            self.conv_id += 1
            x = self.bn(x)
            x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        # if output_stride == 16:
        #     dilations = [1, 6, 12, 18]
        # elif output_stride == 8:
        #     dilations = [1, 12, 24, 36]
        # else:
        #     raise NotImplementedError
        #

        #TI only suppport dilation of 1, 2, 4 and a maximum channel of conv of 1024
        #the new aspp for pld is designed as
        # aspp1 : ch = 256, d = [1]
        # aspp2 : ch = 256, d = [4]
        # aspp3 : ch = 256, d = [2, 4]
        # aspp4 : ch = 256, d = [4, 4]
        inplanes = 512


        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=[0], dilation=[1], BatchNorm=nn.BatchNorm2d)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=[4], dilation=[4], BatchNorm=nn.BatchNorm2d)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=[2, 4], dilation=[2, 4], BatchNorm=nn.BatchNorm2d)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=[4, 4], dilation=[4, 4], BatchNorm=nn.BatchNorm2d)

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
        #                                      BatchNorm(256),
        #                                      nn.ReLU())
        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        #x5 = self.global_avg_pool(x)
        #x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp():
    return ASPP()