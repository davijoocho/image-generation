
import torch 
import numpy as np

N_CHANNELS = 1

VERTICAL_KERNEL_H = 2
VERTICAL_KERNEL_W = 3

HORIZONTAL_KERNEL_H = 1
HORIZONTAL_KERNEL_W = 2

class DualConvBlock(torch.nn.Module):
    def __init__(self, in_channel_dim, out_channel_dim, vertical_pad, horizontal_pad):
        super().__init__()

        self.vertical_conv = torch.nn.Sequential(
            vertical_pad,
            torch.nn.Conv2d(in_channels=in_channel_dim, out_channels=out_channel_dim, kernel_size=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W)),
            torch.nn.BatchNorm2d(num_features=out_channel_dim, affine=False),
            torch.nn.ReLU()
        )

        self.horizontal_conv = torch.nn.Sequential(
            horizontal_pad,
            torch.nn.Conv2d(in_channels=in_channel_dim, out_channels=out_channel_dim, kernel_size=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W)),
            torch.nn.BatchNorm2d(num_features=out_channel_dim, affine=False),
            torch.nn.ReLU()
        )

        torch.nn.init.kaiming_normal_(self.vertical_conv[1].weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.horizontal_conv[1].weight, nonlinearity="relu")
        
    def forward(self, X):
        X_vertical, X_horizontal = X
        return self.vertical_conv(X_vertical), self.horizontal_conv(X_horizontal)

class GatedResidualBlock(torch.nn.Module):
    def __init__(self, channel_dim, vertical_pad, horizontal_pad):
        super().__init__()
        self.channel_dim  = channel_dim

        self.vertical_conv = torch.nn.Sequential(
            vertical_pad,
            torch.nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim * 2, kernel_size=(VERTICAL_KERNEL_H, VERTICAL_KERNEL_W), padding=0),
            torch.nn.BatchNorm2d(num_features=channel_dim * 2, affine=False)
        )

        self.horizontal_conv = torch.nn.Sequential(
            horizontal_pad,
            torch.nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim * 2, kernel_size=(HORIZONTAL_KERNEL_H, HORIZONTAL_KERNEL_W), padding=0),
            torch.nn.BatchNorm2d(num_features=channel_dim * 2, affine=False)
        )

        self.contextual_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channel_dim * 2, out_channels=channel_dim * 2, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(num_features=channel_dim * 2, affine=False)
        )

        self.residual_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(num_features=channel_dim, affine=False)
        )

        torch.nn.init.xavier_normal_(self.vertical_conv[1].weight)
        torch.nn.init.xavier_normal_(self.horizontal_conv[1].weight)
        torch.nn.init.xavier_normal_(self.contextual_conv[0].weight)
        torch.nn.init.xavier_normal_(self.residual_conv[0].weight)

    def forward(self, X):
        X_vertical, X_horizontal = X

        tanh = torch.nn.Tanh()
        sigmoid = torch.nn.Sigmoid()

        vertical = self.vertical_conv(X_vertical)

        v_signal, v_gate = vertical.split(self.channel_dim , dim=1)
        v_signal = tanh(v_signal)
        v_gate = sigmoid(v_gate)
        X_vertical = torch.mul(v_signal, v_gate)

        horizontal = self.horizontal_conv(X_horizontal)

        X_vertical_context = self.contextual_conv(vertical)
        X_horizontal_w_context = torch.add(horizontal, X_vertical_context)
        h_signal, h_gate = X_horizontal_w_context.split(self.channel_dim, dim=1)
        h_signal = tanh(h_signal)
        h_gate = sigmoid(h_gate)
        X_horizontal_w_context = torch.mul(h_signal, h_gate)

        residual = self.residual_conv(X_horizontal_w_context)
        X_horizontal = torch.add(X_horizontal, residual)

        return X_vertical, X_horizontal

class PixelCNN(torch.nn.Module):
    def __init__(self, n_latents):
        super().__init__()
        self.n_latents = n_latents

        self.vertical_pad_a = torch.nn.ZeroPad2d((VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_H, 0))
        self.horizontal_pad_a = torch.nn.ZeroPad2d((HORIZONTAL_KERNEL_W, 0, 0, 0))

        self.vertical_pad_b = torch.nn.ZeroPad2d((VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_W // 2, VERTICAL_KERNEL_H - 1, 0))
        self.horizontal_pad_b = torch.nn.ZeroPad2d((HORIZONTAL_KERNEL_W - 1, 0, 0, 0))

        self.convs = torch.nn.Sequential(
            DualConvBlock(in_channel_dim=1, out_channel_dim=64, vertical_pad=self.vertical_pad_a, horizontal_pad=self.horizontal_pad_a),
            GatedResidualBlock(channel_dim=64, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b),

            DualConvBlock(in_channel_dim=64, out_channel_dim=128, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b),
            GatedResidualBlock(channel_dim=128, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b),

            DualConvBlock(in_channel_dim=128, out_channel_dim=256, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b),
            GatedResidualBlock(channel_dim=256, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b),

            DualConvBlock(in_channel_dim=256, out_channel_dim=512, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b),
            GatedResidualBlock(channel_dim=512, vertical_pad=self.vertical_pad_b, horizontal_pad=self.horizontal_pad_b)
        )

    def forward(self, X):
        X_norm = X.numpy() / self.n_latents

        X_vertical_crop = torch.tensor(X_norm[:, :, :15, :], dtype=torch.float32)
        X_horizontal_crop = torch.tensor(X_norm[:, :, :, :15], dtype=torch.float32)

        X_vertical, X_horizontal = self.convs((X_vertical_crop, X_horizontal_crop))

        return X_horizontal

