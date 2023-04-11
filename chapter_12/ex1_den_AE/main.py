import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def add_noise(image: np.array, gw=10) -> np.array:
    noise = np.random.normal(0, gw, image.shape)
    return image + noise


def add_noise_torch(image: torch.Tensor, gw=10) -> torch.Tensor:
    noise = torch.randn(image.shape) * gw
    return image + noise


class SimpleAE(nn.Module):
    def __init__(self, indim, latent_dim):
        super(SimpleAE, self).__init__()
        self.input_layer = nn.Linear(indim, latent_dim)
        self.output_layer = nn.Linear(latent_dim, indim)

    def forward(self, x):
        x = self.generate(self.latent_space(x))
        return x

    def latent_space(self, x):
        return self.input_layer(x)

    def generate(self, l):
        return self.output_layer(l)


dataset = FashionMNIST('./data', download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleAE(indim=28 * 28, latent_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.view(img.size(0), -1)
        noisy_img = add_noise_torch(img, gw=0.7)
        output = model(noisy_img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.torch')
AE_img = model(noisy_img).detach()

fig, ax = plt.subplots(2, 3, figsize=(10, 10))
for i in range(6):
    first = plt.subplot(3, 6, i + 1)
    first.imshow(img[i].reshape((28, 28)), aspect=1.5)
    second = plt.subplot(3, 6, i + 1 + 6)
    second.imshow(noisy_img[i].reshape((28, 28)), aspect=1.5)
    third = plt.subplot(3, 6, i + 1 + 12)
    third.imshow(AE_img.detach()[i].reshape((28, 28)), aspect=1.5)
    
plt.savefig("plot.pdf")
plt.show()
