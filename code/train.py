
import os
import time
import torch
import torchvision
import numpy as np
import PIL
from vqvae import VectorQuantizedVAE

N_EPOCHS = 128
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
DECAY_RATE = 0.96
BETA = 0.25
N_LATENTS = 512
EMBEDDING_DIM = 32

if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(128, 128)),
        torchvision.transforms.ToTensor()
    ])
    train = torchvision.datasets.MNIST(root=os.getcwd() + "/data", train=True, download=True, transform=transforms)
    train_ldr = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test = torchvision.datasets.MNIST(root=os.getcwd() + "/data", train=False, download=True, transform=transforms)
    test_ldr = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, num_workers=2, drop_last=True)


    model = VectorQuantizedVAE(N_LATENTS, EMBEDDING_DIM)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
    loss = torch.nn.MSELoss()

    model.train()
    for epoch in range(N_EPOCHS):
        for images, label in train_ldr:
            encoder_out, decoder_in, decoder_out = model(images)

            reconstruction_loss = loss(decoder_out, images)
            quantization_loss = loss(encoder_out.detach(), decoder_in)
            commitment_loss = loss(decoder_in, encoder_out.detach())
            total_loss = reconstruction_loss + quantization_loss + BETA*commitment_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        scheduler.step()

    model.eval()
    test_imgs = iter(test_ldr)

    for _ in range(4):
        image, label = next(test_imgs)
        encoder_out, decoder_in, decoder_out = model(image)
        image = decoder_out.squeeze(dim=0).squeeze(dim=0)
        image = (image.detach().numpy() * 255).astype(np.uint8)
        image = PIL.Image.fromarray(image)
        image_file = open(os.getcwd() + "/data/reconstructions/epoch_" + str(epoch) + "_example_" + str(_) + ".png", "wb")
        image.save(image_file, format="PNG")

    torch.save(model.state_dict(), os.getcwd() + "/model/vqvae_latest_version.pt")

