
import torch 
import torchvision
import numpy as np
from cnn import ConvBlock

VERTICAL_KERNEL_H = 2
VERTICAL_KERNEL_W = 3

HORIZONTAL_KERNEL_H = 1
HORIZONTAL_KERNEL_W = 2

class PixelCNN(torch.nn.Module):
    def __init__(self, n_latents):
        super().__init__()
        self.n_latents = n_latents

        self.vertical_pad_a = torch.nn.ZeroPad2d((VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_H, 0))
        self.horizontal_pad_a = torch.nn.ZeroPad2d((HORIZONTAL_KERNEL_W, 0, 0, 0))

        self.vertical_pad_b = torch.nn.ZeroPad2d((VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_H - 1, 0))
        self.horizontal_pad_b = torch.nn.ZeroPad2d((HORIZONTAL_KERNEL_W - 1, 0, 0, 0))

        self.vertical_convs = torch.nn.ParameterList([
            ConvBlock(in_channel_dim=1, out_channel_dim=32, kernel_dim=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=32, out_channel_dim=64, kernel_dim=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=64, out_channel_dim=128, kernel_dim=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=128, out_channel_dim=256, kernel_dim=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=256, out_channel_dim=512, kernel_dim=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W), padding_dim=0)
        ])

        self.horizontal_convs = torch.nn.ParameterList([
            ConvBlock(in_channel_dim=1, out_channel_dim=32, kernel_dim=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=32, out_channel_dim=64, kernel_dim=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=64, out_channel_dim=128, kernel_dim=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=128, out_channel_dim=256, kernel_dim=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W), padding_dim=0),
            ConvBlock(in_channel_dim=256, out_channel_dim=512, kernel_dim=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W), padding_dim=0)
        ])

    def forward(self, X):
        X = X.numpy() / self.n_latents
        X_vertical_crop = torch.tensor(X[:, :, :15, :], dtype=torch.float32)
        X_horizontal_crop = torch.tensor(X[:, :, :, :15], dtype=torch.float32)

        X_vertical_pad = self.vertical_pad_a(X_vertical_crop)
        X_horizontal_pad = self.horizontal_pad_a(X_horizontal_crop)

        X_vertical = None
        X_horizontal = None

        for vertical_conv, horizontal_conv in zip(self.vertical_convs, self.horizontal_convs):
            X_vertical = vertical_conv(X_vertical_pad)
            X_horizontal = horizontal_conv(X_horizontal_pad) + X_vertical

            X_vertical_pad = self.vertical_pad_b(X_vertical)
            X_horizontal_pad = self.horizontal_pad_b(X_horizontal)

        return X_horizontal 

