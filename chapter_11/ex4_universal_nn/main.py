import torch
from matplotlib import pyplot as plt
from torch import nn
# from torchvision import datasets
# from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Tuning parameters
EPOCHS = 40
batch = 1
learning_rate = 2e-4


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=1):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().double()

sin_range = np.linspace(-np.pi, np.pi, 400).reshape(-1, 1)
sin_curve = np.sin(sin_range)
scaler = MinMaxScaler()
sin_range_sc = scaler.fit_transform(sin_range).reshape(-1, 1)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train = np.asarray([sin_range_sc, sin_curve]).reshape(len(sin_range), 2)
data_iterator = DataLoader(train, batch_size=batch, shuffle=True)
full_iterator = DataLoader(train, batch_size=400, shuffle=True)

cuda = torch.device('cuda')

history = []


for idx in range(EPOCHS):
    for minibatch in data_iterator:
        model.train()
        pred = model(minibatch[:, 0:1])
        loss = loss_fn(pred, minibatch[:, 1:2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for full_iter in full_iterator:
        model.eval()
        pred = model(full_iter[:, 0:1])
        loss = loss_fn(pred, full_iter[:, 1:2])
        history.append(loss.detach().numpy())
        print(f"MAE: {loss}")

# plot history
fig = plt.figure()
plt.plot(history)
plt.title('Mean absolute error loss over epochs')
plt.xlabel('Number of iterations (epochs)')
plt.ylabel('MAE loss')
# plt.show()
file_name = 'plot_mae.png'
plt.savefig(file_name)

# save final MAE
file_name = 'mae.txt'
with open(file_name, 'w') as f:
    f.writelines(f"{history[-1]}")

# save torch model
file_name = 'model.torch'
torch.save(model, file_name)

print("Finished")
