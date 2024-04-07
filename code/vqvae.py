
from cnn import ConvBlock, ResidualBlock, TransposeConvBlock
import torch 

N_CHANNELS = 1

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operations = torch.nn.Sequential(
            # (3, 128, 128)
            ResidualBlock(channel_dim=N_CHANNELS, kernel_dim=5),
            ConvBlock(in_channel_dim=N_CHANNELS, out_channel_dim=8, kernel_dim=5, padding_dim=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (8, 64, 64)
            ResidualBlock(channel_dim=8, kernel_dim=3),
            ConvBlock(in_channel_dim=8, out_channel_dim=16, kernel_dim=3, padding_dim=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (16, 32, 32)
            ResidualBlock(channel_dim=16, kernel_dim=3),
            ConvBlock(in_channel_dim=16, out_channel_dim=32, kernel_dim=3, padding_dim=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            # (32, 16, 16)
        )

    def forward(self, X):
        return self.operations(X)

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operations = torch.nn.Sequential(
            # (32, 16, 16)
            ResidualBlock(channel_dim=32, kernel_dim=3),
            TransposeConvBlock(in_channel_dim=32, out_channel_dim=16, kernel_dim=2),
            # (16, 32, 32)
            ResidualBlock(channel_dim=16, kernel_dim=3),
            TransposeConvBlock(in_channel_dim=16, out_channel_dim=8, kernel_dim=2),
            # (8, 64, 64)
            ResidualBlock(channel_dim=8, kernel_dim=5),
            TransposeConvBlock(in_channel_dim=8, out_channel_dim=N_CHANNELS, kernel_dim=2)
            # (3, 128, 128)
        )

    def forward(self, X):
        return self.operations(X)

class VectorQuantizedVAE(torch.nn.Module):
    def __init__(self, n_latents, embedding_dim):
        super().__init__()
        self.n_latents = n_latents
        self.embedding_dim = embedding_dim

        self.encoder = Encoder()
        self.embeddings = torch.nn.Embedding(num_embeddings=self.n_latents, embedding_dim=self.embedding_dim)
        self.decoder = Decoder()

    def compress(self, X):
        return self.encoder(X)

    def fetch_latent(self, X):
        X_transpose = X.permute(dims=(0,2,3,1))
        latent_embeddings = self.embeddings.weight.expand(16, self.n_latents, self.embedding_dim)

        X_latent = []
        for x in X_transpose:
            distances = torch.cdist(x, latent_embeddings)
            nearest_latents = distances.argmin(dim=2).unsqueeze(dim=0)
            X_latent.append(nearest_latents)
        X_latent = torch.cat(X_latent, dim=0)

        return X_latent

    def reconstruct(self, X):
        return self.decoder(X)

    def forward(self, X):
        encoder_out = self.compress(X)

        latents = self.fetch_latent(encoder_out)
        decoder_in = self.embeddings(latents).permute(dims=(0,3,1,2))

        straight_through_estimate = encoder_out + (decoder_in - encoder_out).detach()
        decoder_out = self.reconstruct(straight_through_estimate)

        return encoder_out, decoder_in, decoder_out
    
