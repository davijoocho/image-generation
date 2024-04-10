
import torch 
import time

N_LATENTS = 512
EMBEDDING_DIM = 64

class ConvBlock(torch.nn.Module): 
    def __init__(self, in_channel_dim, out_channel_dim, kernel_dim):
        super().__init__()
        self.operations = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels = in_channel_dim, 
                out_channels = out_channel_dim, 
                kernel_size = kernel_dim, 
                padding = kernel_dim // 2
            ),
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
            ConvBlock(in_channel_dim, upsample_dim, kernel_dim),
            ConvBlock(upsample_dim, in_channel_dim, kernel_dim)
        )

    def forward(self, X):
        return self.operations(X) + X

class TransposeConvBlock(torch.nn.Module):
    def __init__(self, in_channel_dim, out_channel_dim, kernel_dim, stride_dim, padding_dim):
        super().__init__()
        self.operations = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels = in_channel_dim, 
                out_channels = out_channel_dim, 
                kernel_size = kernel_dim, 
                stride = stride_dim, 
                padding = padding_dim
            ),
            torch.nn.BatchNorm2d(num_features=out_channel_dim, affine=False),
            torch.nn.ReLU()
        )
        torch.nn.init.kaiming_normal_(self.operations[0].weight, a=1e-4)

    def forward(self, X):
        return self.operations(X)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operations = torch.nn.Sequential(
            # (3, 128, 128)
            ConvBlock(in_channel_dim=1, out_channel_dim=32, kernel_dim=5),
            ResidualBlock(in_channel_dim=32, upsample_dim=64, kernel_dim=5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (32, 64, 64)
            ConvBlock(in_channel_dim=32, out_channel_dim=64, kernel_dim=3),
            ResidualBlock(in_channel_dim=64, upsample_dim=64, kernel_dim=3),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (64, 32, 32)
            ResidualBlock(in_channel_dim=64, upsample_dim=64, kernel_dim=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            # (64, 16, 16)
        )

    def forward(self, X):
        return self.operations(X)

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operations = torch.nn.Sequential(
            # (64, 16, 16)
            TransposeConvBlock(in_channel_dim=64, out_channel_dim=32, kernel_dim=4, stride_dim=2, padding_dim=1),
            ResidualBlock(in_channel_dim=32, upsample_dim=64, kernel_dim=3),
            # (32, 32, 32)
            TransposeConvBlock(in_channel_dim=32, out_channel_dim=16, kernel_dim=4, stride_dim=2, padding_dim=1),
            ResidualBlock(in_channel_dim=16, upsample_dim=64, kernel_dim=3),
            # (16, 64, 64)
            TransposeConvBlock(in_channel_dim=16, out_channel_dim=1, kernel_dim=4, stride_dim=2, padding_dim=1),
            ResidualBlock(in_channel_dim=1, upsample_dim=64, kernel_dim=1)
            # (3, 128, 128)
        )

    def forward(self, X):
        return self.operations(X)

class VectorQuantizedVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.embeddings = torch.nn.Embedding(num_embeddings=N_LATENTS, embedding_dim=EMBEDDING_DIM)
        self.decoder = Decoder()

    def compress(self, X):
        return self.encoder(X)

    def fetch_latent(self, X):
        X_transpose = X.permute(dims=(0,2,3,1))
        latent_embeddings = self.embeddings.weight.expand(16, N_LATENTS, EMBEDDING_DIM)

        X_distances = []
        for x in X_transpose:
            distances = torch.cdist(x, latent_embeddings).unsqueeze(dim=0)
            X_distances.append(distances)

        X_distances = torch.cat(X_distances, dim=0)
        X_latents = X_distances.argmin(dim=3)

        return X_latents

    def reconstruct(self, X):
        return self.decoder(X)

    def forward(self, X):
        encoder_out = self.compress(X)
        latents = self.fetch_latent(encoder_out)

        decoder_in = self.embeddings(latents).permute(dims=(0, 3, 1, 2))

        straight_through_estimate = encoder_out + (decoder_in - encoder_out).detach()
        decoder_out = self.reconstruct(straight_through_estimate)

        return encoder_out, decoder_in, decoder_out
    
