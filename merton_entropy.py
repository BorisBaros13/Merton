import pickle
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import math
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd

# Force CUDA device and dtype
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor)
print(DEVICE)
# hyperparameters (both model and data)
d = 10
k = d 
input_size = 2+d
hidden_sizes = [1000]
test_size = 1000
ns = [16]
p = 2
start_rate = 1
num_epochs = 100000
trials = 1
x0 = 0
mu = 0.18
vol = 0.44
lmbda = 1
beta = 100
sigma = beta
bins = dict.fromkeys([f"{n}" for n in ns])
for key in bins:
    bins[key] = pd.DataFrame(columns = [f"{r}" for r in hidden_sizes])
print("Total trials:", trials)
# random initialisations
eta = torch.rand(d, device=DEVICE)
eta = eta / torch.norm(eta)

# generating out-of-sample data
xi = torch.normal(mean=mu, std=math.sqrt(vol), size=(test_size,), device=DEVICE)
Z1_test = (4 * torch.rand((test_size, d), device=DEVICE) - 2)
Z2_test = torch.stack([i * eta for i in xi])
test_data = torch.stack((Z1_test, Z2_test), dim=1)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim).to(DEVICE)
        self.hidden_layer.bias.data.zero_()
        self.hidden_layer.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_dim, output_dim).to(DEVICE)
        self.output_layer.bias.data.zero_()
        self.output_layer.bias.requires_grad = False
        self.hidden_dim = hidden_dim

    def forward(self, x):
        hidden_activations = self.sigmoid(self.hidden_layer(x))
        raw_output = self.output_layer(hidden_activations)
        return raw_output / self.hidden_dim

def regularising(th1, th2, p, r):
    full_th = torch.cat((th1, th2))
    nrm = torch.norm(full_th)
    return (nrm**p + 1e-9 * torch.exp(nrm))

def single_loss(X2, lmbda):
    return -(1 - torch.exp(-lmbda * X2))

def loss(X2, theta_1, theta_2, lmbda, beta):
    loss_vmap = torch.vmap(single_loss, in_dims=(0, None))
    ells = loss_vmap(X2, lmbda)
    L = torch.mean(ells) * theta_1.shape[0]
    theta_vmap = torch.vmap(regularising, in_dims=(0, 1, None, None))
    ells_by_j = theta_vmap(theta_1, theta_2, p, theta_1.shape[0])
    L_regularising = torch.sum(ells_by_j) / (2 * beta ** 2)
    pseudo_loss = L + L_regularising
    return L, pseudo_loss, ells

def ce(lmbda, v):
    return torch.log(1 + v) / lmbda

# Training loop with exponential decay scheduler
print("Entropy-Regularised ERM")
initial_lr = 0.1
final_lr = 0.00001
gamma = (final_lr / initial_lr) ** (1 / num_epochs)

for trial in range(trials):
    print("Trial: ", trial + 1)
    for n in ns:
        print("n: ", n)
        xi = torch.normal(mean=mu, std=math.sqrt(vol), size=(n,), device=DEVICE)
        Z1 = torch.rand((n, d), device=DEVICE) - 0.5
        Z2 = torch.stack([i * eta for i in xi])
        data = torch.stack((Z1, Z2), dim=1)
        for r in hidden_sizes:
            print("r: ", r)
            model = NeuralNet(input_size, r, k).to(DEVICE)
            optimiser = torch.optim.AdamW(model.parameters(), lr=initial_lr)
            scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

            for epoch in range(num_epochs):
                # if epoch % 10000 == 0:
                #     print(epoch)
                learning_rate = optimiser.param_groups[0]['lr']
                optimiser.zero_grad()
                X0 = torch.zeros(n, device=DEVICE)
                X1 = torch.matmul(torch.ones(d, device=DEVICE), data[:, 0, :].T) / d
                X1 = X1.view(n, 1)
                U1 = model(torch.cat((data[:, 1, :], X1, torch.ones(n, 1, device=DEVICE)), dim=1))
                dotter = torch.vmap(torch.dot, in_dims=(0, 0))
                X2 = X1 + dotter(U1, data[:, 1, :]).view(n, 1)
                actual_loss, pseudo_loss, ells_by_sample = loss(X2, model.hidden_layer.weight, model.output_layer.weight, lmbda, beta)
                pseudo_loss.backward()
                optimiser.step()
                scheduler.step()

                if epoch != num_epochs - 1:
                    with torch.no_grad():
                        model.hidden_layer.weight += torch.sqrt(torch.tensor(learning_rate, device=DEVICE)) * sigma / beta * torch.randn((r, 1), device=DEVICE)
                        model.output_layer.weight += torch.sqrt(torch.tensor(learning_rate, device=DEVICE)) * sigma / beta * torch.randn((k, r), device=DEVICE)
                else:
                    with torch.no_grad():
                        X0_test = torch.ones(test_size, device=DEVICE)
                        X1_test = torch.matmul(torch.ones(d, device=DEVICE), test_data[:, 0, :].T) / d
                        X1_test = X1_test.view(test_size, 1)
                        U1_test = model(torch.cat((test_data[:, 1, :], X1_test, torch.ones(test_size, 1, device=DEVICE)), dim=1))
                        dotter = torch.vmap(torch.dot, in_dims=(0, 0))
                        X2_test = X1_test + dotter(U1_test, test_data[:, 1, :]).view(test_size, 1)
                        test_loss, pseudo_test_loss, test_ells_by_sample = loss(X2_test, model.hidden_layer.weight, model.output_layer.weight, lmbda, beta)
                        bins[f"{n}"].loc[trial, f"{r}"] = torch.abs(actual_loss - test_loss).detach().item() / r

bins.to_pickle("entropy_diffs.pkl")
