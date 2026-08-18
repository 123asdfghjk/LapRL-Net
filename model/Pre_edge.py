import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.sobel_x = nn.Conv2d(in_channels, in_channels,
                                 kernel_size=3, padding=1,
                                 groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels,
                                 kernel_size=3, padding=1,
                                 groups=in_channels, bias=False)

        self._init_sobel_weights()

        for param in self.sobel_x.parameters():
            param.requires_grad = False
        for param in self.sobel_y.parameters():
            param.requires_grad = False

        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def _init_sobel_weights(self):

        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=torch.float32)

        with torch.no_grad():
            for i in range(self.in_channels):
                self.sobel_x.weight[i, 0] = sobel_kernel_x
                self.sobel_y.weight[i, 0] = sobel_kernel_y

    def forward(self, x):

        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)

        grad_mag = torch.cat([grad_x, grad_y], dim=1)

        edge_weights = torch.sigmoid(self.conv1x1(grad_mag))
        return x * edge_weights


class BoundaryAwareBlock(nn.Module):

    def __init__(self, channels_list, out_channels=18):
        super(BoundaryAwareBlock, self).__init__()
        self.channels_list = channels_list

        self.conv_s1 = nn.Conv2d(channels_list[0], out_channels, kernel_size=1)
        self.conv_s2 = nn.Conv2d(channels_list[1], out_channels, kernel_size=1)
        self.conv_s3 = nn.Conv2d(channels_list[2], out_channels, kernel_size=1)
        self.conv_s4 = nn.Conv2d(channels_list[3], out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, s1, s2, s3, s4):
        s1_processed = self.conv_s1(s1)
        s2_processed = self.conv_s2(s2)
        s2_upsampled = F.interpolate(s2_processed, size=s1.shape[2:],
                                     mode='bilinear', align_corners=False)

        s3_processed = self.conv_s3(s3)
        s3_upsampled = F.interpolate(s3_processed, size=s1.shape[2:],
                                     mode='bilinear', align_corners=False)
        s4_processed = self.conv_s4(s4)
        s4_upsampled = F.interpolate(s4_processed, size=s1.shape[2:],
                                     mode='bilinear', align_corners=False)

        combined = torch.cat([s1_processed, s2_upsampled, s3_upsampled, s4_upsampled], dim=1)

        x = self.relu(self.bn1(self.conv1(combined)))
        x = self.relu(self.bn2(self.conv2(x)))
        edge_pred = torch.sigmoid(self.conv3(x))

        return edge_pred

class EGRM(nn.Module):
    def __init__(self):
        super(EGRM, self).__init__()

        channels_list = [9, 9, 18, 36]

        self.sobel_s1 = SobelLayer(channels_list[0])
        self.sobel_s2 = SobelLayer(channels_list[1])
        self.sobel_s3 = SobelLayer(channels_list[2])
        self.sobel_s4 = SobelLayer(channels_list[3])

        self.bab = BoundaryAwareBlock(channels_list, out_channels=18)

    def forward(self, features):

        s1, s2, s3, s4 = features

        s1_sobel = self.sobel_s1(s1)
        s2_sobel = self.sobel_s2(s2)
        s3_sobel = self.sobel_s3(s3)
        s4_sobel = self.sobel_s4(s4)

        edge_pred = self.bab(s1_sobel, s2_sobel, s3_sobel, s4_sobel)

        return edge_pred