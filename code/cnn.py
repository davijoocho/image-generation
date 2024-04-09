
import torch

class ConvBlock(torch.nn.Module): 
    def __init__(self, in_channel_dim, out_channel_dim, kernel_dim, padding_dim):
        super().__init__()
        self.operations = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel_dim, out_channels=out_channel_dim, kernel_size=kernel_dim, padding=padding_dim),
            torch.nn.BatchNorm2d(num_features=out_channel_dim, affine=False),
            torch.nn.ReLU()
        )
        torch.nn.init.kaiming_normal_(self.operations[0].weight, a=1e-4)

    def forward(self, X):
        return self.operations(X)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel_dim, upsample_dim, kernel_dim):
        super().__init__()
        self.operations = torch.nn.Sequential(
            ConvBlock(in_channel_dim, upsample_dim, kernel_dim, kernel_dim // 2),
            ConvBlock(upsample_dim, in_channel_dim, kernel_dim, kernel_dim // 2)
        )

    def forward(self, X):
        return self.operations(X) + X

class TransposeConvBlock(torch.nn.Module):
    def __init__(self, in_channel_dim, out_channel_dim, kernel_dim):
        super().__init__()
        self.operations = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channel_dim, out_channels=out_channel_dim, kernel_size=kernel_dim, stride=kernel_dim),
            torch.nn.BatchNorm2d(num_features=out_channel_dim, affine=False),
            torch.nn.ReLU()
        )
        torch.nn.init.kaiming_normal_(self.operations[0].weight, a=1e-4)

    def forward(self, X):
        return self.operations(X)

