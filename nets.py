from torch import nn
from utils import get_outputpadding
import torch


class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 128, 5, 3, 1),  # n,128,32,32
            nn.LeakyReLU(),
            ConvolutionalLayer(128, 256, 4, 2, 1),  # n,256,16,16
            ConvolutionalLayer(256, 512, 4, 2, 1),  # n,512,8,8
            ConvolutionalLayer(512, 1024, 4, 2, 1),  # n,1024,4,4
            nn.Conv2d(1024, 1, 4),  # n,1,1,1
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.layer(data)
        return output


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            ConvTransposeLayer(128, 1024, 4, 1, 0, get_outputpadding(1, 4, 4, 1, 0)),  # n,1024,4,4
            ConvTransposeLayer(1024, 512, 4, 2, 1, get_outputpadding(4, 8, 4, 2, 1)),  # n,512,8,8
            ConvTransposeLayer(512, 256, 4, 2, 1, get_outputpadding(8, 16, 4, 2, 1)),  # n,256,16,16
            ConvTransposeLayer(256, 128, 4, 2, 1, get_outputpadding(16, 32, 4, 2, 1)),  # n,128,32,32
            nn.ConvTranspose2d(128, 3, 5, 3, 1, get_outputpadding(32, 96, 5, 3, 1)),  # n,1,96,96
            nn.Tanh(),
        )

    def forward(self, data):
        output = self.layer(data)
        return output


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, data):
        return self.layer(data)


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, data):
        return self.layer(data)


if __name__ == '__main__':
    d_input = torch.Tensor(1, 1, 96, 96)
    g_input = torch.randn(1, 128, 1, 1)
    g_net = GNet()
    d_net = DNet()
    g_out = g_net(g_input)
    d_out = d_net(d_input)
    print(g_out.shape)
    print(d_out.shape)
