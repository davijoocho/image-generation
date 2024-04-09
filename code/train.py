
import os
import time
import torch
import torchvision
import numpy as np
import PIL
from vqvae import VectorQuantizedVAE
from pixelcnn import PixelCNN

N_EPOCHS = 128
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
DECAY_RATE = 0.96
BETA = 0.25
N_LATENTS = 512
EMBEDDING_DIM = 64

if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(128, 128)),
        torchvision.transforms.ToTensor()
    ])

    train = torchvision.datasets.MNIST(root=os.getcwd() + "/data", train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    test = torchvision.datasets.MNIST(root=os.getcwd() + "/data", train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, num_workers=2, drop_last=True)

    vqvae = VectorQuantizedVAE(N_LATENTS, EMBEDDING_DIM)
    vqvae_optimizer = torch.optim.SGD(vqvae.parameters(), lr=LEARNING_RATE)
    vqvae_scheduler = torch.optim.lr_scheduler.ExponentialLR(vqvae_optimizer, gamma=DECAY_RATE)
    mse_loss = torch.nn.MSELoss()

    pixelcnn = PixelCNN(N_LATENTS)
    pixelcnn_optimizer = torch.optim.SGD(pixelcnn.parameters(), lr=LEARNING_RATE)
    pixelcnn_scheduler = torch.optim.lr_scheduler.ExponentialLR(pixelcnn_optimizer, gamma=DECAY_RATE)
    nll_loss = torch.nn.NLLLoss()
    log_softmax = torch.nn.LogSoftmax(dim=1)

    for epoch in range(N_EPOCHS):
        vqvae.train()
        for images, labels in train_loader:
            encoder_out, decoder_in, decoder_out = vqvae(images)

            reconstruction_loss = mse_loss(decoder_out, images)
            quantization_loss = mse_loss(encoder_out.detach(), decoder_in)
            commitment_loss = mse_loss(decoder_in, encoder_out.detach())
            vqvae_loss = reconstruction_loss + quantization_loss + BETA * commitment_loss

            vqvae_optimizer.zero_grad()
            vqvae_loss.backward()
            vqvae_optimizer.step()

        vqvae_scheduler.step()

        vqvae.eval()
        test_imgs = iter(test_loader)
        for _ in range(4):
            image, label = next(test_imgs)
            encoder_out, decoder_in, decoder_out = vqvae(image)
            image = decoder_out.squeeze(dim=0).squeeze(dim=0)
            image = (image.detach().numpy() * 255).astype(np.uint8)
            image = PIL.Image.fromarray(image)
            image_file = open(os.getcwd() + "/data/reconstructions/epoch_" + str(epoch) + "_example_" + str(_) + ".png", "wb")
            image.save(image_file, format="PNG")

    pixelcnn.train()
    for epoch in range(N_EPOCHS):
        for images, labels in train_loader:
            latent = vqvae.fetch_latent(vqvae.compress(images))
            logits = pixelcnn(latent.unsqueeze(dim=1))
            log_proba = log_softmax(logits)
            pixelcnn_loss = nll_loss(log_proba, latent)

            pixelcnn_optimizer.zero_grad()
            pixelcnn_loss.backward()
            pixelcnn_optimizer.step()

    torch.save(vqvae.state_dict(), os.getcwd() + "/model/vqvae_latest_version.pt")
    torch.save(pixelcnn.state_dict(), os.getcwd() + "/model/pixelcnn_latest_version.pt")

