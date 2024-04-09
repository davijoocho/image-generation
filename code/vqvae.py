
import torch 

N_CHANNELS = 1

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

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.operations = torch.nn.Sequential(
            # (3, 128, 128)
            ResidualBlock(in_channel_dim=N_CHANNELS, upsample_dim=128, kernel_dim=5),
            ConvBlock(in_channel_dim=N_CHANNELS, out_channel_dim=128, kernel_dim=5, padding_dim=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (128, 64, 64)
            ResidualBlock(in_channel_dim=128, upsample_dim=256, kernel_dim=3),
            ConvBlock(in_channel_dim=128, out_channel_dim=256, kernel_dim=3, padding_dim=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # (256, 32, 32)
            ResidualBlock(in_channel_dim=256, upsample_dim=512, kernel_dim=3),
            ConvBlock(in_channel_dim=256, out_channel_dim=64, kernel_dim=3, padding_dim=1),
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
            ResidualBlock(in_channel_dim=64, upsample_dim=128, kernel_dim=3),
            TransposeConvBlock(in_channel_dim=64, out_channel_dim=128, kernel_dim=2),
            # (128, 32, 32)
            ResidualBlock(in_channel_dim=128, upsample_dim=256, kernel_dim=3),
            TransposeConvBlock(in_channel_dim=128, out_channel_dim=256, kernel_dim=2),
            # (256, 64, 64)
            ResidualBlock(in_channel_dim=256, upsample_dim=512, kernel_dim=5),
            TransposeConvBlock(in_channel_dim=256, out_channel_dim=N_CHANNELS, kernel_dim=2)
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
    
