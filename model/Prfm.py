import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
from math import sqrt
from .seg_hrnet_config import get_hrnet_cfg
import torch
import math
from torch.nn.functional import upsample
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, \
    PairwiseDistance
from torch.nn import functional as F
from torch.autograd import Variable

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Unfold):
            pass
        elif isinstance(m, GELU):
            pass
        elif isinstance(m, Softmax):
            pass
        elif isinstance(m, Sigmoid):
            pass
        else:
            m.initialize()

class LOCAL(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=2):
        super().__init__()
        mid_channels = inplanes // reduction1
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
    def initialize(self):
        weight_init(self)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, max_heads=4, reduction_ratio=4):
        super().__init__()
        heads = max_heads
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.reduction_ratio = reduction_ratio
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads, r = self.heads, self.reduction_ratio

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        k, v = map(lambda t: reduce(t, 'b c (h r1) (w r2) -> b c h w', 'mean', r1=r, r2=r), (k, v))

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        return self.to_out(out)
    def initialize(self):
        weight_init(self)


class MixFeedForward(nn.Module):
    def __init__(self, dim, expansion_factor):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)
    def initialize(self):
        weight_init(self)


class GLOBAL(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction1=2, patch_size=3):
        super(GLOBAL, self).__init__()
        mid_channels = inplanes // reduction1
        self.reductionLayers = nn.Sequential(
            nn.Conv2d(inplanes, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.get_overlap_patches = nn.Unfold(patch_size, dilation=1, stride=2, padding=1)
        patch_dim = mid_channels * patch_size * patch_size

        self.overlap_embed = nn.Conv2d(patch_dim, planes, kernel_size=1)
        self.SelfAttention = EfficientSelfAttention(dim=planes, max_heads=4, reduction_ratio=4)
        self.ffd = MixFeedForward(dim=planes, expansion_factor=2)
        self.LN = nn.InstanceNorm2d(planes, affine=True)

    def forward(self, x):
        x_size = x.size()
        x = self.reductionLayers(x)
        h, w = x.shape[-2:]

        x = self.get_overlap_patches(x)
        num_patches = x.shape[-1]
        ratio = int(sqrt((h * w) / num_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)
        x = self.overlap_embed(x)

        x = self.SelfAttention(self.LN(x)) + x
        x = self.ffd(self.LN(x)) + x
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x
    def initialize(self):
        weight_init(self)

class DFEM(nn.Module):
    def __init__(self, inc, outc):
        super(DFEM, self).__init__()

        self.Conv_1 = nn.Sequential(
            nn.Conv2d(inc * 2, outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        self.Conv = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, diff, accom):
        cat = torch.cat([accom, diff], dim=1)
        cat = self.Conv_1(cat) + diff + accom
        c = self.Conv(cat) + cat
        c = self.relu(c) + diff
        return c

    def initialize(self):
        weight_init(self)

class PRFM4(nn.Module):
    def __init__(self, inplanes=144, planes=36, BatchNorm=nn.BatchNorm2d):
        super(PRFM4,self).__init__()
        out_size = inplanes // 4
        self.LOCAL = LOCAL(inplanes, out_size, BatchNorm, reduction1=2)
        self.GLOBAL = GLOBAL(inplanes, out_size, BatchNorm, reduction1=2, patch_size=3)
        self.merge_context = DFEM(out_size, planes)
    def forward(self, x):
        return self.merge_context(self.LOCAL(x), self.GLOBAL(x))

class PRFM3(nn.Module):
    def __init__(self, inplanes=72, planes=18, BatchNorm=nn.BatchNorm2d):
        super(PRFM3,self).__init__()
        out_size = inplanes // 4
        self.LOCAL = LOCAL(inplanes, out_size, BatchNorm, reduction1=2)
        self.GLOBAL = GLOBAL(inplanes, out_size, BatchNorm, reduction1=2, patch_size=3)
        self.merge_context = DFEM(out_size, planes)

    def forward(self, x):
        return self.merge_context(self.LOCAL(x), self.GLOBAL(x))

class PRFM2(nn.Module):
    def __init__(self, inplanes=36, planes=9, BatchNorm=nn.BatchNorm2d):
        super(PRFM2,self).__init__()
        out_size = max(9, inplanes // 4)
        self.LOCAL = LOCAL(inplanes, out_size, BatchNorm, reduction1=2)
        self.GLOBAL = GLOBAL(inplanes, out_size, BatchNorm, reduction1=2, patch_size=3)
        self.merge_context = DFEM(out_size, planes)

    def forward(self, x):
        return self.merge_context(self.LOCAL(x), self.GLOBAL(x))
class PRFM1(nn.Module):
    def __init__(self, inplanes=18, planes=9, BatchNorm=nn.BatchNorm2d):
        super(PRFM1,self).__init__()
        out_size = max(9, inplanes // 2)
        self.LOCAL = LOCAL(inplanes, out_size, BatchNorm, reduction1=2)
        self.GLOBAL = GLOBAL(inplanes, out_size, BatchNorm, reduction1=2, patch_size=3)
        self.merge_context = DFEM(out_size, planes)

    def forward(self, x):
        return self.merge_context(self.LOCAL(x), self.GLOBAL(x))

class NLCDetection(nn.Module):
    def __init__(self, args):
        super(NLCDetection, self).__init__()

        self.crop_size = args['crop_size']
        FENet_cfg = get_hrnet_cfg()
        num_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']
        feat1_num, feat2_num, feat3_num, feat4_num = num_channels

        self.prfm4 = PRFM4(feat4_num, feat4_num//4, nn.BatchNorm2d)
        self.prfm3 = PRFM3(feat3_num, feat3_num//4, nn.BatchNorm2d)
        self.prfm2 = PRFM2(feat2_num, feat2_num//4, nn.BatchNorm2d)
        self.prfm1 = PRFM1(feat1_num, feat1_num//2, nn.BatchNorm2d)

        self.mask_gen4 = nn.Conv2d(feat4_num//4, 1, kernel_size=3, padding=1)
        self.mask_gen3 = nn.Conv2d(feat3_num//4, 1, kernel_size=3, padding=1)
        self.mask_gen2 = nn.Conv2d(feat2_num//4, 1, kernel_size=3, padding=1)
        self.mask_gen1 = nn.Conv2d(feat1_num//2, 1, kernel_size=3, padding=1)

    def forward(self, feat):
        s1, s2, s3, s4 = feat
        if s1.shape[2:] == self.crop_size:
            pass
        else:
            s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)

        z4 = self.prfm4(s4)
        mask4 = self.mask_gen4(z4)
        mask4U = F.interpolate(mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

        s3 = s3 * mask4U
        z3 = self.prfm3(s3)
        mask3 = self.mask_gen3(z3)
        mask3U = F.interpolate(mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

        s2 = s2 * mask3U
        z2 = self.prfm2(s2)
        mask2 = self.mask_gen2(z2)
        mask2U = F.interpolate(mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

        s1 = s1 * mask2U
        z1 = self.prfm1(s1)
        mask1 = self.mask_gen1(z1)

        prfm_feat = [z1, z2, z3, z4]
        return mask1, mask2, mask3, mask4, prfm_feat

    def initialize(self):
        weight_init(self)