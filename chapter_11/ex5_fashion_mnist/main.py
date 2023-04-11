import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


class CNN_fashion(nn.Module):
    def __init__(self) -> None:
        super(CNN_fashion, self).__init__()

        # initialize first set of cnn layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.2)

        # initialize second set of cnn layers
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.25)

        # initialize output layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=100 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # first conv pass
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # second conv pass
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # dense layers and output layer
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Training
    print(" Initializing the model and starting training... ")

    train_set = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    INIT_LR = 10e-4
    BATCH_SIZE = 32
    EPOCHS = 15
    train_size = 48000

    criterion = nn.CrossEntropyLoss()
    model = CNN_fashion()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)

    val_size = len(train_set) - train_size
    train_data, val_data = torch.utils.data.random_split(train_set, (train_size, val_size))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)

    train_losses = []
    val_losses = []

    for epoch in tqdm.trange(EPOCHS):
        model.train()

        train_loss = 0
        train_accuracy = 0

        for (x, y) in train_loader:
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_accuracy = 0

            for (x, y) in val_loader:
                pred = model(x)
                epoch_val_loss += criterion(pred, y).item()
                epoch_val_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

        epoch_train_loss = train_loss / (train_size / BATCH_SIZE)
        epoch_train_accuracy = train_accuracy / train_size
        epoch_val_loss = epoch_val_loss / (val_size / BATCH_SIZE)
        epoch_val_accuracy = epoch_val_accuracy / val_size

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"\n Epoch {epoch}\n\
                Training loss: {epoch_train_loss}\n\
                Training accuracy: {epoch_train_accuracy}\n\
                Validation loss: {epoch_val_loss}\n\
                Validation accuracy: {epoch_val_accuracy}\n"
        )


    # test

    test_data = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000)
    model.eval()
    correct_preds = 0

    for (x, y) in test_loader:
        pred = model(x)
        correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_accuracy = correct_preds / 10000
    print(f"Test accuracy: {test_accuracy}")

    compiled = torch.jit.script(model)
    compiled.save("model.torch")

    # plots
    fig = plt.figure(figsize=(14, 10))
    fig_grid = GridSpec(2, 1)

    fig1 = fig.add_subfigure(fig_grid[0, 0])
    ax0 = fig1.add_subplot(1, 1, 1)
    ax0.plot(range(EPOCHS), train_losses, c="navy", label="Training loss")
    ax0.plot(range(EPOCHS), val_losses, c="purple", label="Validation loss")
    ax0.set_title(f"Learning curve, Test accuracy: {test_accuracy}")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Cross-entropy loss")
    ax0.legend(loc="upper right")

    fig2 = fig.add_subfigure(fig_grid[1, 0])
    grid = GridSpec(4, 6)

    np_test_img = test_data.data.detach().numpy()
    np_test_labels = test_data.targets.detach().numpy()
    pred = pred.detach().numpy()

    for i in range(4):
        for j in range(6):
            ax = fig2.add_subplot(grid[i, j])
            ax.imshow(np.reshape(np_test_img[i * 10 + j], (28, 28)) * 255, cmap="pink")
            ax.set_title(f"Pred: {np.argmax(pred[i*10+j])}, True: {np_test_labels[i*10+j]}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.savefig("plot.pdf")
