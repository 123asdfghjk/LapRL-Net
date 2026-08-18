import torch
import torch.nn as nn
from torch.nn import functional as F
from .GaussianSmooth import GaussianSmooth

class LRAR(nn.Module):
    def __init__(self,in_c,gauss_ker_size=3,scale=[2,4,8],drop_rate=0.2):
        super(LRAR, self).__init__()
        self.scale = scale
        self.gauss_ker_size = gauss_ker_size
        self.smoothing = nn.ModuleDict()
        for s in self.scale:
            self.smoothing['scale-'+str(s)] = GaussianSmoothing(in_c, self.gauss_ker_size, s)
        self.scale_weights = nn.Parameter(torch.ones(len(scale)))
        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_c*len(scale), in_c,
                                                kernel_size=1, stride=1,
                                                bias=False,groups=1),
                                                nn.BatchNorm2d(in_c),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=drop_rate)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def down(self,x,s):
        return F.interpolate(x,scale_factor=s,
                             mode='bilinear',
                             align_corners=False)
    def up (self,x, size):
        return F.interpolate(x,size=size,mode='bilinear',align_corners=False)

    def forward(self, x):
        H, W = x.shape[2:]
        diff_list = []

        for i, s in enumerate(self.scale):
            sm = self.smoothing[f'scale-{s}'](x)
            sm = self.down(sm, 1 / s)
            sm = self.up(sm, (H, W))

            F_dif = self.scale_weights[i] * (x - sm)
            diff_list.append(F_dif)

        diff = torch.cat(diff_list, dim=1)

        return self.conv_1x1(diff)