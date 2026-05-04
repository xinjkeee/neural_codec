import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            WNConv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(channels, channels, kernel_size=1)
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=512, embed_dim=128):
        super().__init__()
        self.init_conv = WNConv1d(in_channels, 64, kernel_size=7, padding=3)

        self.down1 = nn.Sequential(
            WNConv1d(64, 128, kernel_size=8, stride=2, padding=3),
            ResBlock1D(128),
            ResBlock1D(128),
        )
        self.down2 = nn.Sequential(
            WNConv1d(128, 256, kernel_size=8, stride=4, padding=2),
            ResBlock1D(256),
            ResBlock1D(256),
        )
        self.down3 = nn.Sequential(
            WNConv1d(256, 512, kernel_size=8, stride=5, padding=2),
            ResBlock1D(512),
            ResBlock1D(512),
        )
        self.down4 = nn.Sequential(
            WNConv1d(512, hidden_dim, kernel_size=8, stride=4, padding=2),
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim),
        )
        self.final_conv = WNConv1d(hidden_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        z = self.final_conv(x)
        return z

class Decoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=512):
        super().__init__()
        self.init_conv = WNConv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)

        self.up4 = nn.Sequential(
            ResBlock1D(hidden_dim),
            ResBlock1D(hidden_dim),
            WNConvTranspose1d(hidden_dim, 512, kernel_size=8, stride=4, padding=2),
        )
        self.up3 = nn.Sequential(
            ResBlock1D(512),
            ResBlock1D(512),
            WNConvTranspose1d(512, 256, kernel_size=8, stride=5, padding=2, output_padding=1),
        )  # output_padding=1 для точного размера при stride=5
        self.up2 = nn.Sequential(
            ResBlock1D(256),
            ResBlock1D(256),
            WNConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
        )
        self.up1 = nn.Sequential(
            ResBlock1D(128),
            ResBlock1D(128),
            WNConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),
        )
        self.final_conv = WNConv1d(64, 1, kernel_size=7, padding=3)

    def forward(self, z):
        x = self.init_conv(z)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.final_conv(x)
        return x